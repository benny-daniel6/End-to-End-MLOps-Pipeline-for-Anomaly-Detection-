import numpy as np
import pandas as pd
import os

def gen_data(n_samples=5000,contamination=0.01,filepath="data/transactions.csv"):
    print("Generating synthetic transaction data...")

    os.makedirs(os.path.dirname(filepath),exist_ok=True)

    n_normal= int(n_samples * (1 - contamination))
    normal_amounts=np.random.lognormal(mean=3, sigma=0.5, size=n_normal)
    normal_times=np.random.normal(loc=12, scale=4, size=n_normal)%24
    normal_data=np.vstack((normal_amounts, normal_times)).T

    n_anomalies=int(n_samples * contamination)
    anomaly_amounts=np.random.uniform(low=500, high=2000, size=n_anomalies)
    anomaly_times=np.random.uniform(low=0, high=24, size=n_anomalies)
    anomaly_data=np.vstack((anomaly_amounts, anomaly_times)).T

    x= np.vstack((normal_data, anomaly_data))
    y= np.array([0] * n_normal + [1] * n_anomalies)
    shuffled_indices=np.random.permutation(len(x))
    x=x[shuffled_indices]
    y=y[shuffled_indices]
    df=pd.DataFrame(x, columns=["Transaction_Amount", "TimeOfDay"])
    df['isAnamoly'] = y

    df.to_csv(filepath, index=False)
    print(f"Data generated and saved to {filepath}.")
if __name__ == "__main__":
    gen_data()

