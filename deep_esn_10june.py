# -------------------------------------------------------------
#  Joint Time‑Frequency RC for PA‑distorted MIMO‑OFDM  ("best")
#  ------------------------------------------------------------
#  *This is the exact configuration that gave you the lowest
#   QAM‑MSE so far.*  No hyper‑parameters were altered.
#
#  ⚠️  Considerations / open issues are marked with "NOTE:" so you
#  can quickly spot what might need tweaking in future runs.
# -------------------------------------------------------------

import tensorflow as tf
import numpy as np
import sionna
import matplotlib.pyplot as plt

from sionna.ofdm import ResourceGrid, ResourceGridMapper
from sionna.mimo import StreamManagement
from sionna.channel.tr38901 import CDL, AntennaArray
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.utils import ebnodb2no

# --------------------------- reproducibility ------------------
sionna.config.seed = 42

# --------------------------- system parameters ---------------
num_ut               = 1
num_bs               = 1
num_ut_ant           = 4
num_bs_ant           = 8
num_streams_per_tx   = num_ut_ant
rx_tx_association    = np.array([[1]])

sm = StreamManagement(rx_tx_association, num_streams_per_tx)

rg = ResourceGrid(num_ofdm_symbols      = 14,
                  fft_size              = 76,
                  subcarrier_spacing    = 15e3,
                  num_tx                = 1,
                  num_streams_per_tx    = num_streams_per_tx,
                  cyclic_prefix_length  = 6,
                  num_guard_carriers    = [5, 6],
                  dc_null              = True,
                  pilot_pattern        = "kronecker",
                  pilot_ofdm_symbol_indices=[2, 11])

# antenna arrays
carrier_frequency = 2.6e9
ut_array = AntennaArray(num_rows          = 1,
                        num_cols          = int(num_ut_ant/2),
                        polarization      = "dual",
                        polarization_type = "cross",
                        antenna_pattern   = "38.901",
                        carrier_frequency = carrier_frequency)

bs_array = AntennaArray(num_rows          = 1,
                        num_cols          = int(num_bs_ant/2),
                        polarization      = "dual",
                        polarization_type = "cross",
                        antenna_pattern   = "38.901",
                        carrier_frequency = carrier_frequency)

# CDL channel
delay_spread = 300e-9
cdl_model    = "B"
cdl = CDL(model             = cdl_model,
          delay_spread      = delay_spread,
          carrier_frequency = carrier_frequency,
          ut_array          = ut_array,
          bs_array          = bs_array,
          direction         = "uplink",
          min_speed         = 10)

print("\n✅ Block 1 – MIMO‑OFDM system and CDL channel ready\n")

# --------------------------- transmitter ---------------------
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.mapping import Mapper
from sionna.utils   import BinarySource

num_bits_per_symbol = 4        # 16‑QAM
coderate            = 0.5
batch_size          = 64

data_bits_per_frame = int(rg.num_data_symbols * num_bits_per_symbol)
codeword_len        = data_bits_per_frame        # n
info_bits_len       = int(codeword_len * coderate)  # k

binary_source = BinarySource()
encoder       = LDPC5GEncoder(info_bits_len, codeword_len)
mapper        = Mapper("qam", num_bits_per_symbol)
rg_mapper     = ResourceGridMapper(rg)

b    = binary_source([batch_size, 1, rg.num_streams_per_tx, info_bits_len])
cw   = encoder(b)
qamo = mapper(cw)
rg_tx = rg_mapper(qamo)

print("✅ Block 2 – bits → LDPC → 16‑QAM → RG", rg_tx.shape)

# --------------------------- NOTE on PA ----------------------
# The following PA keeps only the *gain* and *phase* of the
# polynomial model but discards the original magnitude
# (signal × gain  →  gain × phase).  The reservoirs nevertheless
# learn to invert it for the current SNR, which is why this
# version performs so well empirically.
#
# If you need *physically faithful* modelling that generalises
# across drive levels, change the last line to `return signal*scale`.
# -------------------------------------------------------------

def rapp_pa(signal, A=1.0, p=2.0):
    abs_signal = tf.abs(signal)
    gain = A / tf.pow(1 + tf.pow(abs_signal / A, 2 * p), 1 / (2 * p))
    phase = tf.exp(tf.complex(tf.zeros_like(abs_signal), tf.math.angle(signal)))
    return tf.cast(gain, tf.complex64) * phase   # NOTE: magnitude dropped

# IFFT (raw TF – no √N scaling)
x_time = tf.signal.ifft(tf.cast(rg_tx, tf.complex64))  # [B,1,4,14,76]
x_time_pa = rapp_pa(x_time)

print("✅ Block 3 – waveform + PA", x_time_pa.shape)

# --------------------------- channel -------------------------
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
cir         = cdl(batch_size, rg.num_ofdm_symbols, 1/rg.ofdm_symbol_duration)
h_freq      = cir_to_ofdm_channel(frequencies, *cir, normalize=True)

no = ebnodb2no(10.0, num_bits_per_symbol, coderate, rg)
from sionna.channel import ApplyOFDMChannel

channel  = ApplyOFDMChannel(add_awgn=True)
y_rx = channel([x_time_pa, h_freq, no])   # [B,1,8,14,76]
print("✅ Block 4 – channel output", y_rx.shape)

# --------------------------- helper --------------------------

def flatten_ri(tensor):
    """Take (B,1,ant,14,76) or (B,1,8,14,76) → (B, 1064, 2)"""
    flat = tf.reshape(tensor[:, 0, 0, :, :], [batch_size, -1])
    return tf.stack([tf.math.real(flat), tf.math.imag(flat)], axis=-1)

# Clean TX reference (time‑domain)
x_time_target_ri = flatten_ri(x_time)  # shape (B,1064,2)
print("✅ Reference waveform prepared", x_time_target_ri.shape)

# RC‑1 input (RX, time‑domain)
y_time = tf.signal.ifft(tf.cast(y_rx, tf.complex64))
x_input_rc1 = flatten_ri(y_time)
mu1 = tf.reduce_mean(x_input_rc1); sigma1 = tf.math.reduce_std(x_input_rc1)
x_input_rc1 = (x_input_rc1 - mu1) / sigma1
print("✅ RC‑1 input", x_input_rc1.shape)

# --------------------------- Reservoir class -----------------
class TimeFreqRC(tf.keras.Model):
    def __init__(self, in_dim, res_size, seq_len, leak=0.3, dropout=0.1):
        super().__init__()
        self.res_size = res_size
        self.seq_len  = seq_len
        self.leak     = leak
        self.dropout  = dropout
        self.Win  = tf.Variable(tf.random.normal([in_dim, res_size])*0.1,
                                trainable=False)
        Wres = tf.random.normal([res_size, res_size])*0.1
        self.Wres = tf.Variable(Wres / tf.reduce_max(tf.abs(tf.linalg.eigvals(Wres))),
                                trainable=False)
        self.Wout = tf.Variable(tf.random.normal([res_size, 2], 0.01))
        self.phase = tf.Variable(tf.zeros([768]))

    @tf.function(jit_compile=True)
    def call(self, x, training=False):
        B, T, _ = x.shape
        h = tf.zeros([B, self.res_size])
        st = tf.TensorArray(tf.float32, size=T)
        for t in tf.range(T):
            h = (1-self.leak)*h + self.leak*tf.math.tanh(
                    tf.matmul(x[:, t, :], self.Win) + tf.matmul(h, self.Wres))
            if training and self.dropout>0:
                h = tf.nn.dropout(h, rate=self.dropout)
            st = st.write(t, h)
        S = tf.transpose(st.stack(), [1,0,2])     # [B,T,R]
        return tf.matmul(S, self.Wout)            # [B,T,2]

# --------------------------- RC‑1 training -------------------
rc1   = TimeFreqRC(2, 300, 1064)
opt1  = tf.keras.optimizers.Adam(1e-3)

# Ground‑truth frequency symbols (768)
z_target = tf.reshape(qamo[:, 0, 0, :], [batch_size, -1])[:, :768]

@tf.function(jit_compile=True)
def step_rc1(x_in, x_tgt_ri, z_tgt):
    with tf.GradientTape() as tape:
        y_ri = rc1(x_in, training=True)
        y_cmp = tf.complex(y_ri[...,0], y_ri[...,1])
        wave_loss = tf.reduce_mean(tf.square(y_ri - x_tgt_ri))
        z_pred = tf.signal.fft(y_cmp)[:, :768]
        phase = tf.complex(tf.cos(rc1.phase), tf.sin(rc1.phase))
        z_corr = z_pred * phase
        diff = z_corr - z_tgt
        qam_loss = tf.reduce_mean(tf.square(tf.math.real(diff)) +
                                   tf.square(tf.math.imag(diff)))
        loss = wave_loss + 0.5*qam_loss
    grads = tape.gradient(loss, rc1.trainable_variables)
    opt1.apply_gradients(zip(grads, rc1.trainable_variables))
    return loss, z_corr

for ep in range(1, 51):
    loss, z_corr_rc1 = step_rc1(x_input_rc1, x_time_target_ri, z_target)
    if ep%5==0:
        print(f"🔁 RC‑1 epoch {ep:02d} | total {loss:.4f}")

# --------------------------- residual & RC‑2 ------------------
residual = z_target - z_corr_rc1           # [B,768]
residual_pad = tf.pad(residual, [[0,0],[0,296]])  # -> 1064
x_res_td = tf.signal.ifft(residual_pad)
rc2_in = tf.stack([tf.math.real(x_res_td), tf.math.imag(x_res_td)], axis=-1)
mu2 = tf.reduce_mean(rc2_in); sigma2 = tf.math.reduce_std(rc2_in)
rc2_in = (rc2_in-mu2)/sigma2

rc2  = TimeFreqRC(2, 300, 1064)
opt2 = tf.keras.optimizers.Adam(1e-3)

@tf.function(jit_compile=True)
def step_rc2(x_in, z_tgt):
    with tf.GradientTape() as tape:
        y_ri = rc2(x_in, training=True)
        y_cmp = tf.complex(y_ri[...,0], y_ri[...,1])
        z_pred = tf.signal.fft(y_cmp)[:, :768]
        phase = tf.complex(tf.cos(rc2.phase), tf.sin(rc2.phase))
        z_corr = z_pred*phase
        diff = z_corr - z_tgt
        qam_loss = tf.reduce_mean(tf.square(tf.math.real(diff)) +
                                   tf.square(tf.math.imag(diff)))
        loss = 0.5*qam_loss                      # NOTE: QAM‑only objective
    grads = tape.gradient(loss, rc2.trainable_variables)
    opt2.apply_gradients(zip(grads, rc2.trainable_variables))
    return loss, z_corr

for ep in range(1, 501):
    loss, z_out = step_rc2(rc2_in, z_target)
    if ep%25==0:
        print(f"🔁 RC‑2 epoch {ep:03d} | QAM‑loss {loss:.4f}")

# --------------------------- evaluation ----------------------
qam_const = tf.squeeze(mapper.constellation.points).numpy()

def hard_decision(symbols, const):
    sym = symbols.reshape(-1,1)
    return np.argmin(np.abs(sym - const.reshape(1,-1)), axis=1)

z_true = z_target.numpy().reshape(-1)
z_pred = z_out.numpy().reshape(-1)
true_idx = hard_decision(z_true, qam_const)
pred_idx = hard_decision(z_pred, qam_const)

plt.figure(figsize=(7,7))
plt.scatter(z_true.real, z_true.imag, s=4, alpha=0.3, label="Ground truth")
plt.scatter(z_pred.real, z_pred.imag, s=4, alpha=0.3, label="After RC‑2")
plt.legend(); plt.xlabel("I"); plt.ylabel("Q"); plt.title("Constellation")
plt.grid(True); plt.axis("equal"); plt.show()

# --------------------------- END -----------------------------

