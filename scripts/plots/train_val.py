import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = '/mnt/c/Users/fdebe/Documents/EPI/Models/chat-ptolemaic/runs/2026-03-12_16-18-00_gpt110m_general_pretrain_v2/metrics.csv'
df = pd.read_csv(path)

mask_val = ~df.val_loss.isna()

# smoothing window
window = 400

train_smooth = df.train_loss.rolling(window, min_periods=1).mean()
val_smooth = df.val_loss.rolling(window*5, min_periods=1).mean()

plt.plot(df.step,df.step*0+3.2,':',color='k')
plt.plot(df.step,df.step*0+3.1,':',color='k')
plt.plot(df.step,df.step*0+3.0,':',color='k')
plt.plot(df.step,df.step*0+2.9,':',color='k')
plt.plot(df.step, train_smooth, label="train")
plt.plot(df.step[mask_val], val_smooth[mask_val], label="val")

plt.legend()
plt.savefig('train_val_loss.png')