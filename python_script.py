import pandas as pd
import numpy as np
from funkcijas import sataisit_grafiku
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
from sklearn.utils.class_weight import compute_class_weight
from datetime import date, timedelta
pd.set_option("display.precision", 2)


# šī daļa ir brīvi maināma, bet šobrīd ir saregulēta uz maksimāli labāko precizitāti
dienas = 1
pieaugums = 2  # %
custom_treshold = 0.53
robezas_datums = '2025-11-01'
rolling_for_max = 12
papildus_svars_tiem_kas_mazak = 0.27
stochastic_range = 24
stochastic_smooter = 3
min_max_extra_jutiba = 1
# ----------------------------------------------------------------------------------

# popular_crypto =['BTC-USD','ETH-USD','SOL-USD','ADA-USD','BNB-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'LTC-USD']
ticker = "BTC-USD"  # or "BTC-USD", "SOL1-USD", etc.
df = yf.Ticker(ticker).history(start=(date.today() - timedelta(days=700)).strftime('%Y-%m-%d'), # yahoo stundas datiem maksimums ir 730 dienas
                               end=date.today().strftime('%Y-%m-%d'), interval="1h").reset_index()
df['Close'] = df['Close'].astype(float)

df['Future_Close'] = df['Close'].shift(-dienas*24).rolling(rolling_for_max).max()
df['Target_up'] = ((df['Future_Close'] / df['Close']) >= 1+pieaugums/100).astype(int)
df['price_up_down'] = np.where(df['Close'] > df['Close'].rolling(rolling_for_max).mean(), 'up', 'down')

# papildus_svars_tiem_kas_mazak = round(int(df.Target_up.value_counts()[1]) / int(df.Target_up.value_counts()[0]),2)

cik_dienas_jau_uz_leju, down_skaits, cik_dienas_jau_uz_augsu, up_skaits = [], 0, [], 0 
for d in df['price_up_down']:
    down_skaits = 0 if d=='down' else down_skaits + 1
    cik_dienas_jau_uz_leju.append(down_skaits)
    up_skaits = 0 if d=='up' else up_skaits + 1
    cik_dienas_jau_uz_augsu.append(up_skaits)
df['cik_dienas_jau_uz_leju'] = cik_dienas_jau_uz_leju
df['cik_dienas_jau_uz_augsu'] = cik_dienas_jau_uz_augsu
df['uz_augsu_streak_delta'] = (df['cik_dienas_jau_uz_augsu'] - df['cik_dienas_jau_uz_augsu'].shift(1))
df['uz_leju_streak_delta'] = (df['cik_dienas_jau_uz_leju'] - df['cik_dienas_jau_uz_leju'].shift(1))


is_local_max = df['Close'] == df['Close'].rolling(48).max()
df['hours_since_high'] = is_local_max[::-1].cumsum()[::-1]


df['High_Low'] = (df['High'] - df['Low']) / df['Low'] # stundas diapazons %
df['Close_Open'] = (df['Close'] - df['Open']) / df['Open'] # noslēguma/atvēruma %
for i in [6, 12, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 364, 512]:
    df[f'Return_{i}'] = df['Close'].pct_change(i) # Returns
    df[f'SMA_{i}'] = df['Close'].rolling(i).mean() # simple moving average
    df[f'EMA_{i}'] = df['Close'].ewm(span=i, adjust=False).mean() # exponential moving average

# df[f'Stochastic_{stochastic_range}'] = (100 * (df['Close'] - df['Low'].rolling(stochastic_range).min()) / 
#                                         (df['High'].rolling(stochastic_range).max() - df['Low'].rolling(stochastic_range).min()))
# df[f'Stochastic_smooth_{stochastic_range}'] = df[f'Stochastic_{stochastic_range}'].rolling(stochastic_smooter).mean()


df[f'Vol_std_{i}'] = df['Close'].pct_change().rolling(24).std() # Volatility regime
df[f'Vol_std_{i}'] = df['Close'].pct_change().rolling(168).std() # Volatility regime

# Candle anatomy
df['Body'] = (df['Close'] - df['Open']).abs()
df['UpperWick'] = df['High'] - df[['Open','Close']].max(axis=1)
df['LowerWick'] = df[['Open','Close']].min(axis=1) - df['Low']

# RSI - visu padara sliktāku :/

features = [col for col in df.columns if 'price_up_down'not in col] # izmetam kolonas, kas nav skaitļi
for not_needed in ['Datetime', 'Future_Close', 'Future_Close_d', 'Target_up','Volume','Return_6','Close_Open','Dividends','Stock Splits']:
    features = [col for col in features if not_needed not in col]

X_train = df.loc[df['Datetime'] <= robezas_datums, features]
y_train = df.loc[df['Datetime'] <= robezas_datums, 'Target_up']

X_test  = df.loc[df['Datetime'] > robezas_datums, features]
y_test  = df.loc[df['Datetime'] > robezas_datums, 'Target_up']

weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=y_train)
class_weight={0:1, 1:1+papildus_svars_tiem_kas_mazak}

model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=50,
                               max_features='sqrt', class_weight=class_weight, random_state=42)

model.fit(X_train, y_train)
predicted_prob = model.predict_proba(X_test)[:, 1]

prognozes_varbutibas = pd.DataFrame({'prognoze':list(predicted_prob)})
prognozes_varbutibas['max_vid'] = prognozes_varbutibas['prognoze'].rolling(rolling_izmers).max() / prognozes_varbutibas['prognoze'].rolling(rolling_izmers).min() - prognozes_varbutibas['prognoze']
prognozes_varbutibas['prognoze'] = np.where(prognozes_varbutibas['max_vid']>min_max_extra_jutiba,
                                            prognozes_varbutibas['prognoze']+prognozes_varbutibas['max_vid'],prognozes_varbutibas['prognoze'])

# y_pred = (predicted_prob >= custom_treshold).astype(int) # Apply custom threshold
y_pred = (np.where(prognozes_varbutibas['prognoze'] >= custom_treshold,1,0)).astype(int) # Apply custom threshold

df_test = df.loc[y_test.index].copy()
df_test['Predicted_up'] = y_pred

try:
    ieprieksejais = round(float(up_report['0'][2]+up_report['1'][1]+up_report['accuracy'][2]),3)
except:
    ieprieksejais = '?'
up_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).iloc[:3,:3].reset_index()
up_report = up_report.rename(columns={'index':'UP_report'})

print(f'''Pirms tam bija {ieprieksejais}\nkopā {round(float(up_report['0'][2]+up_report['1'][1]),3)} pie {round(float(up_report['accuracy'][2]),3)} precizitātes = {round(float(up_report['0'][2]+up_report['1'][1]+up_report['accuracy'][2]),3)}''')
print(up_report)
sataisit_grafiku(df_test, df_test.Datetime, df_test['Close'], ticker, pieaugums, dienas)
