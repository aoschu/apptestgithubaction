# -------------------------------------------------------------
#  Joint Time‑Frequency RC for PA‑distorted MIMO‑OFDM  ("best")
#  ------------------------------------------------------------
#  *Server‑ready, streaming version (completed)*
#  ─────────────────────────────────────────────
#  • Same hyper‑parameters that produced your lowest QAM‑MSE.
#  • Infinite synthetic data via tf.data; total training time is
#    controlled only by TOTAL_STEPS_RC1 / RC2.
#  • Automatic single‑ or multi‑GPU via MirroredStrategy.
#  • RC‑2 trains on QAM‑MSE only (fast convergence).
#  • PA model still drops signal magnitude (to replicate winning run).
# -------------------------------------------------------------

import os, tensorflow as tf, numpy as np, sionna

# ─────────────────────────────────────────────────────────────
# 0)  Device placement
# ─────────────────────────────────────────────────────────────

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
if gpus:
    STRATEGY = (tf.distribute.MirroredStrategy() if len(gpus) > 1
                else tf.distribute.get_strategy())
    DEVICE_STR = f"{len(gpus)} GPU(s)"
else:
    STRATEGY = tf.distribute.get_strategy()
    DEVICE_STR = "CPU"
print(f"✅ Runtime device: {DEVICE_STR}")

# ─────────────────────────────────────────────────────────────
# 1)  Reproducibility
# ─────────────────────────────────────────────────────────────

sionna.config.seed = 42
tf.random.set_seed(42)

# ─────────────────────────────────────────────────────────────
# 2)  Hyper‑parameters (unchanged)
# ─────────────────────────────────────────────────────────────

BATCH_SIZE          = 64             # per replica
NUM_BITS_PER_SYMBOL = 4              # 16‑QAM
CODERATE            = 0.5
EBNO_MIN, EBNO_MAX  = 10.0, 10.0     # fixed Eb/N0 (set range to sweep)
TOTAL_STEPS_RC1     = 50
TOTAL_STEPS_RC2     = 500

NUM_UT, NUM_BS      = 1, 1
NUM_UT_ANT, NUM_BS_ANT = 4, 8
NUM_STREAMS         = NUM_UT_ANT

# ─────────────────────────────────────────────────────────────
# 3)  Static Sionna objects
# ─────────────────────────────────────────────────────────────

from sionna.ofdm import ResourceGrid, ResourceGridMapper
from sionna.mimo import StreamManagement
from sionna.channel.tr38901 import CDL, AntennaArray
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.utils import ebnodb2no

RX_TX_ASSOC = np.array([[1]])
SM = StreamManagement(RX_TX_ASSOC, NUM_STREAMS)

RG = ResourceGrid(num_ofdm_symbols=14, fft_size=76, subcarrier_spacing=15e3,
                  num_tx=1, num_streams_per_tx=NUM_STREAMS,
                  cyclic_prefix_length=6, num_guard_carriers=[5,6], dc_null=True,
                  pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2,11])

CARRIER_FREQUENCY = 2.6e9
UT_ARR = AntennaArray(num_rows=1, num_cols=NUM_UT_ANT//2, polarization="dual",
                      polarization_type="cross", antenna_pattern="38.901",
                      carrier_frequency=CARRIER_FREQUENCY)
BS_ARR = AntennaArray(num_rows=1, num_cols=NUM_BS_ANT//2, polarization="dual",
                      polarization_type="cross", antenna_pattern="38.901",
                      carrier_frequency=CARRIER_FREQUENCY)

CDL_CHANNEL = CDL(model="B", delay_spread=300e-9, carrier_frequency=CARRIER_FREQUENCY,
                  ut_array=UT_ARR, bs_array=BS_ARR, direction="uplink", min_speed=10)

FREQS  = subcarrier_frequencies(RG.fft_size, RG.subcarrier_spacing)
N_DATA = int(RG.num_data_symbols * NUM_BITS_PER_SYMBOL)
K_DATA = int(N_DATA * CODERATE)

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.mapping import Mapper
from sionna.utils import BinarySource
from sionna.channel import ApplyOFDMChannel

ENCODER   = LDPC5GEncoder(K_DATA, N_DATA)
MAPPER    = Mapper("qam", NUM_BITS_PER_SYMBOL)
RG_MAPPER = ResourceGridMapper(RG)
BINARY    = BinarySource()
CHANNEL   = ApplyOFDMChannel(add_awgn=True)
CONSTELLATION = tf.squeeze(MAPPER.constellation.points)  # (16,)

# ─────────────────────────────────────────────────────────────
# 4)  Data generator (infinite)
# ─────────────────────────────────────────────────────────────

def rapp_pa(signal, A=1.0, p=2.0):
    """RAPP PA that *drops magnitude* (matches winning run)."""
    abs_signal = tf.abs(signal)
    gain = tf.pow(1 + tf.pow(abs_signal/A, 2*p), -1/(2*p))
    return tf.cast(gain, tf.complex64) * tf.exp(1j*tf.math.angle(signal))

PAD_ZEROS = 1064 - 768  # 296‑sample pad to match RC‑2 logic

@tf.function
def prepare_batch():
    # bits → LDPC → QAM → RG
    b  = BINARY([BATCH_SIZE, 1, NUM_STREAMS, K_DATA])
    c  = ENCODER(b)
    x  = MAPPER(c)
    x_rg = RG_MAPPER(x)
    # TX time‑domain
    x_time = tf.signal.ifft(tf.cast(x_rg, tf.complex64))
    x_time_pa = rapp_pa(x_time)
    # channel
    cir = CDL_CHANNEL(BATCH_SIZE, RG.num_ofdm_symbols, 1/RG.ofdm_symbol_duration)
    h_f = cir_to_ofdm_channel(FREQS, *cir, normalize=True)
    ebno_db = tf.random.uniform([], EBNO_MIN, EBNO_MAX)
    no = ebnodb2no(ebno_db, NUM_BITS_PER_SYMBOL, CODERATE, RG)
    y_rc = CHANNEL([x_time_pa, h_f, no])
    # flatten RX
    y_td = tf.signal.ifft(tf.cast(y_rc, tf.complex64))
    y_flat = tf.reshape(y_td[:,0,0,:,:], [BATCH_SIZE, -1])  # (B,1064)
    x_in = tf.stack([tf.math.real(y_flat), tf.math.imag(y_flat)], axis=-1)
    x_in = (x_in - tf.reduce_mean(x_in)) / tf.math.reduce_std(x_in)
    # targets
    x_time_flat = tf.reshape(x_time[:,0,0,:,:], [BATCH_SIZE, -1])
    x_tgt = tf.stack([tf.math.real(x_time_flat), tf.math.imag(x_time_flat)], axis=-1)
    z_tgt = tf.reshape(x[:,0,0,:], [BATCH_SIZE, -1])  # (B,768)
    return x_in, x_tgt, z_tgt

def gen():
    while True:
        yield prepare_batch()

global_batch = BATCH_SIZE * STRATEGY.num_replicas_in_sync
print(f"✅ tf.data stream ready — global batch = {global_batch}")

ds = (tf.data.Dataset.from_generator(gen,
        output_signature=(tf.TensorSpec((None,1064,2), tf.float32),
                          tf.TensorSpec((None,1064,2), tf.float32),
                          tf.TensorSpec((None,768),    tf.complex64)))
      .prefetch(tf.data.AUTOTUNE))

# ─────────────────────────────────────────────────────────────
# 5)  Reservoir definition
# ─────────────────────────────────────────────────────────────

class TimeFreqRC(tf.keras.Model):
    def __init__(self, R):
        super().__init__()
        self.R = R
        Win = tf.random.normal([2, R])*0.1
        Wres = tf.random.normal([R, R])*0.1
        Wres /= tf.reduce_max(tf.abs(tf.linalg.eigvals(Wres)))
        self.Win  = tf.Variable(Win, trainable=False)
        self.Wres = tf.Variable(Wres, trainable=False)
        self.Wout = tf.Variable(tf.random.normal([R, 2], 0.01))
        self.phase= tf.Variable(tf.zeros([768]))
    @tf.function(jit_compile=True)
    def call(self, x, training=False):
        B,T,_ = x.shape
        h = tf.zeros([B, self.R])
        states = tf.TensorArray(tf.float32, size=T)
        for t in tf.range(T):
            h = 0.7*h + 0.3*tf.math.tanh(tf.matmul(x[:,t,:], self.Win) + tf.matmul(h, self.Wres))
            states = states.write(t, h)
        return tf.matmul(tf.transpose(states.stack(), [1,0,2]), self.Wout)

# ─────────────────────────────────────────────────────────────
# 6)  Distributed models & optimisers
# ─────────────────────────────────────────────────────────────

with STRATEGY.scope():
    rc1, rc2 = TimeFreqRC(300), TimeFreqRC(300)
    opt1, opt2 = tf.keras.optimizers.Adam(1e-3), tf.keras.optimizers.Adam(1e-3)

# training step helpers
@tf.function(jit_compile=True)
def step_rc1(x_in, x_tgt, z_tgt):
    with tf.GradientTape() as tape:
        y = rc1(x_in, True)
        y_cmp = tf.complex(y[...,0], y[...,1])
        loss_wave = tf.reduce_mean(tf.square(y - x_tgt))
        z_pred = tf.signal.fft(y_cmp)[:, :768]
        z_corr = z_pred * tf.complex(tf.cos(rc1.phase), tf.sin(rc1.phase))
        diff = z_corr - z_tgt
        loss_qam = tf.reduce_mean(tf.square(tf.math.real(diff))+tf.square(tf.math.imag(diff)))
        loss = loss_wave + 0.5*loss_qam
    opt1.apply_gradients(zip(tape.gradient(loss, rc1.trainable_variables), rc1.trainable_variables))
    residual = z_tgt - z_corr
    return loss, loss_qam, residual

@tf.function(jit_compile=True)
def step_rc2(residual, x_tgt, z_tgt):
    residual_pad = tf.pad(residual, [[0,0],[0,PAD_ZEROS]])
    x_res_time = tf.signal.ifft(residual_pad)
    x_in = tf.stack([tf.math.real(x_res_time), tf.math.imag(x_res_time)], axis=-1)
    x_in = (x_in - tf.reduce_mean(x_in)) / tf.math.reduce_std(x_in)
    with tf.GradientTape() as tape:
        y = rc
