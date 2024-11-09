![Screenshot 2024-11-09 093527](https://github.com/user-attachments/assets/587b89b8-564d-4396-bd00-cf66ffc97d35)## PROBLEM STATEMENT - 2
# AI-Driven Dynamic Public Transportation Scheduler

Efficiently managing public transportation in urban areas is a significant challenge, especially with fluctuating commuter demand, traffic variations, and unexpected events. This project aims to develop an AI-driven platform that autonomously schedules and dispatches public transport vehicles, ensuring dynamic adaptability and optimized efficiency.

## Problem Statement

As urban populations grow, managing public transportation to meet dynamic demand becomes more complex. Traditional scheduling systems often struggle to adapt to real-time fluctuations in commuter demand, traffic conditions, or unscheduled events like concerts, sports matches, and road closures. This can result in:

- Overcrowded buses or trains.
- Under-utilized vehicles.
- Longer commute times and delays.

These inefficiencies impact both commuters and transportation authorities, necessitating an intelligent, adaptive system for real-time scheduling and routing.

## Solution Overview
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import tensorflow as tf
import os
!pip install pyngrok
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")


from flask import Flask, render_template_string, request
from pyngrok import ngrok
from pyngrok import ngrok

# Set the auth token
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
Ensure required directory exists
os.makedirs('/content', exist_ok=True)
df= pd.read_csv("/content/TRAFFIC_DATA.csv")
df
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
df.set_index('timestamp', inplace=True)
df.info()
df.head()


# Check for missing values and fill them with forward-fill method
df.fillna(method='ffill', inplace=True)


features = ['vehicle_count', 'avg_speed', 'traffic_density', 'event', 'crowd_size',
            'weather_condition', 'temperature', 'road_condition', 'zone',
            'latitude', 'longitude', 'altitude', 'road_width']

# Separate categorical and numerical features, including geographic and road width
categorical_features = ['event', 'weather_condition', 'road_condition']
numerical_features = ['vehicle_count', 'avg_speed', 'traffic_density', 'crowd_size',
                      'temperature', 'latitude', 'longitude', 'altitude', 'road_width']


encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cats = encoder.fit_transform(df[categorical_features])


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_numerical = scaler.fit_transform(df[numerical_features])


data = np.concatenate([scaled_numerical, encoded_cats], axis=1)


def get_max_vehicle_capacity(road_width):
    # Assume each meter of road width can accommodate up to 2 vehicles
    return road_width * 2


def create_sequences_with_capacity(data, seq_length, road_widths):
    sequences = []
    labels = []
    capacities = []
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i])
        labels.append(data[i][0])  # vehicle count as the label
        capacities.append(get_max_vehicle_capacity(road_widths[i]))  # Calculate capacity
    return np.array(sequences), np.array(labels), np.array(capacities)


road_widths = df['road_width'].values
SEQ_LENGTH = 60
X, y, max_capacities = create_sequences_with_capacity(data, SEQ_LENGTH, road_widths)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


predictions = model.predict(X_test)

predicted_congestion_rescaled = scaler.inverse_transform(
    np.concatenate((predictions, X_test[:, -1, 1:len(numerical_features)]), axis=1)
)[:, 0]


predicted_congestion_adjusted = np.minimum(predicted_congestion_rescaled, max_capacities[:len(predicted_congestion_rescaled)])

zones = df['zone'].unique()


rerouted_zones = [zones[(int(cong) + 1) % len(zones)] if cong > max_capacities[i]
                  else zones[int(cong) % len(zones)]
                  for i, cong in enumerate(predicted_congestion_adjusted)]


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Convert index (timestamp) to list to access it easily
timestamps = df.index.tolist()
current_timestamp = None
# Displays Congestion & Rerouting details for the first 24 hours
for i in range(24*6):
    if timestamps[i] != current_timestamp:
        current_timestamp = timestamps[i]
        print(f"\nTimestamp: {current_timestamp.strftime('%Y/%m/%d %H:%M')}")

    print(f"Predicted Congestion: {predicted_congestion_adjusted[i]:<15.2f}, "
          f"Current Zone: {df['zone'][i]:<10}, "
          f"Suggested Rerouting Zone: {rerouted_zones[i]:<15}")


predictions_df = pd.DataFrame({
    'timestamp': df.index[:len(predicted_congestion_adjusted)],
    'congestion_level': predicted_congestion_adjusted / max_capacities[:len(predicted_congestion_adjusted)],  # Normalized congestion level
    'zone': df['zone'][:len(predicted_congestion_adjusted)],
    'rerouted_zone': rerouted_zones[:len(predicted_congestion_adjusted)],
    'max_capacity': max_capacities[:len(predicted_congestion_adjusted)],
    'vehicle_count_before': df['vehicle_count'][:len(predicted_congestion_adjusted)],
    'vehicle_count_after': np.minimum(predicted_congestion_adjusted, max_capacities[:len(predicted_congestion_adjusted)]),
})


predictions_csv_path = '/content/prediction.csv'  # Specify your desired path
predictions_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")


df['rerouted_zones'] = np.nan  # Initialize with NaN
df.iloc[-len(rerouted_zones):, df.columns.get_loc('rerouted_zones')] = rerouted_zones


before_rerouting_data = df.groupby('zone')[['vehicle_count', 'latitude', 'longitude', 'altitude']].mean().reset_index()
after_rerouting_data = df.groupby('rerouted_zones')[['vehicle_count', 'latitude', 'longitude', 'altitude']].mean().reset_index()


print(before_rerouting_data.head())
print(after_rerouting_data.head())


kmeans_before = KMeans(n_clusters=3, random_state=42)
before_rerouting_data['cluster'] = kmeans_before.fit_predict(before_rerouting_data[['vehicle_count', 'latitude', 'longitude', 'altitude']])


plt.figure(figsize=(10, 6))
sns.scatterplot(x=before_rerouting_data['zone'],
                y=before_rerouting_data['vehicle_count'],
                hue=before_rerouting_data['cluster'],
                palette='Set1', s=100)
plt.xlabel("Zones")
plt.ylabel("Vehicle Count")
plt.title("Clustering Before Rerouting")
plt.xticks(rotation=90)
plt.show()


if len(rerouted_zones) < len(df):
    rerouted_zones = np.resize(rerouted_zones, len(df))
elif len(rerouted_zones) > len(df):
    rerouted_zones = rerouted_zones[:len(df)]


df['rerouted_zone'] = rerouted_zones


after_rerouting_data = df.groupby('rerouted_zone')['vehicle_count'].mean().reset_index()


if len(after_rerouting_data) >= 3:
    kmeans_after = KMeans(n_clusters=3, random_state=42)
    after_rerouting_data['cluster'] = kmeans_after.fit_predict(after_rerouting_data[['vehicle_count']])
else:
    print(f"Not enough samples for KMeans clustering. Available samples: {len(after_rerouting_data)}")
    after_rerouting_data['cluster'] = np.zeros(len(after_rerouting_data))  # Assign all to one cluster if not enough samples


plt.figure(figsize=(10, 6))
sns.scatterplot(x=after_rerouting_data['rerouted_zone'],
                y=after_rerouting_data['vehicle_count'],
                hue=after_rerouting_data['cluster'],
                palette='Set2', s=100)
plt.xlabel("Zones")
plt.ylabel("Vehicle Count")
plt.title("Clustering After Rerouting")
plt.xticks(rotation=90)
plt.show()


before_rerouting_data = df.groupby('zone')['vehicle_count'].mean().reset_index()
after_rerouting_data = df.groupby('rerouted_zone')['vehicle_count'].mean().reset_index()


comparison_data = before_rerouting_data.merge(after_rerouting_data, left_on='zone', right_on='rerouted_zone', suffixes=('_before', '_after'))


plt.figure(figsize=(10, 6))
for zone in comparison_data['zone']:
    zone_data = comparison_data[comparison_data['zone'] == zone]
    plt.plot(['Before', 'After'],
             [zone_data['vehicle_count_before'].values[0], zone_data['vehicle_count_after'].values[0]],
             marker='o', label=f'Zone {zone}')
plt.title("Congestion Diversion Across Zones Before and After Rerouting")
plt.xlabel("Condition")
plt.ylabel("Average Vehicle Count")
plt.legend(title='Zone')


plt.figure(figsize=(12, 8))
traffic_pivot = df.pivot_table(values='vehicle_count', index='zone', columns=df.index.hour, aggfunc='mean')
sns.heatmap(traffic_pivot, cmap='YlOrRd')
plt.xlabel('Hour of the Day')
plt.ylabel('Zone')
plt.title('Traffic Volume Heatmap by Zone and Time')
plt.show()


comparison_data.set_index('zone', inplace=True)
comparison_data[['vehicle_count_before', 'vehicle_count_after']].plot(kind='bar', figsize=(12, 6))
plt.xlabel('Zone')
plt.ylabel('Average Vehicle Count')
plt.title('Traffic Volume Before and After Rerouting by Zone')
plt.legend(['Before Rerouting', 'After Rerouting'])
plt.show()


def generate_traffic_report(timestamp, congestion_level, zone, rerouted_zone, max_capacity, vehicle_count_before, vehicle_count_after):
    report = f"""
    Traffic Report:
    At {timestamp}, the congestion level in {zone} is {congestion_level*100:.1f}%.
    Rerouting is suggested to {rerouted_zone}. The maximum road capacity is {max_capacity} vehicles.
    Before rerouting, there were {vehicle_count_before} vehicles, and after rerouting, there are {vehicle_count_after} vehicles.
    """
    return report


for index, row in predictions_df.head(10).iterrows():
    timestamp = row['timestamp']
    congestion_level = row['congestion_level']
    zone = row['zone']
    rerouted_zone = row['rerouted_zone']
    max_capacity = row['max_capacity']
    vehicle_count_before = row['vehicle_count_before']
    vehicle_count_after = row['vehicle_count_after']

  
    traffic_report = generate_traffic_report(timestamp, congestion_level, zone, rerouted_zone, max_capacity, vehicle_count_before, vehicle_count_after)
    print(f"Traffic Report for {timestamp}:\n{traffic_report}\n")ine feature sets









### Deliverables

Develop an AI-based platform that autonomously manages the scheduling, routing, and dispatching of public transportation vehicles (buses, trains, etc.) based on real-time data and predictive insights. The system should:

- **Predict Commuter Demand:** Leverage historical and live data to forecast demand across routes.
- **Adapt Scheduling in Real-Time:** Respond to live traffic conditions and events by dynamically adjusting schedules.
- **Optimize Routing and Dispatching:** Minimize congestion, reduce wait times, and balance vehicle utilization.

### Objectives

1. **Real-Time Commuter Demand Prediction**  
   - Use AI/ML algorithms to predict commuter demand based on historical patterns, seasonal trends, and real-time data (e.g., weather, local events, traffic data).
   
2. **Dynamic Scheduling and Routing**  
   - Continuously adapt transport schedules and routes in response to live traffic conditions and known events (e.g., concerts, sporting events, road closures).
   
3. **Optimization of Dispatching**  
   - Maximize vehicle utilization by balancing commuter loads, reducing underutilized runs, and minimizing commuter wait times.
   - Reduce congestion and ensure equitable coverage across urban regions.









## Key Features

-**LSTM-Based Time Series Forecasting:
Purpose: Predicts traffic patterns by analyzing historical traffic data over time.
-**Description: LSTM networks, a type of RNN, are ideal for sequential data like traffic volume. They learn from past patterns (e.g., rush-hour peaks) to accurately predict future congestion levels

---
##OUTPUT

![Screenshot 2024-11-09 093805](https://github.com/user-attachments/assets/fc442efc-faf5-453b-ad56-69938b0adbee)
![Screenshot 2024-11-09 093837](https://github.com/user-attachments/assets/6cf6e3a1-bda7-4a39-a9a4-53886cf79d3a)
![Screenshot 2024-11-09 093859](https://github.com/user-attachments/assets/f50e68cc-fbd4-4d71-82dd-45d288cd7000)
![Screenshot 2024-11-09 093920](https://github.com/user-attachments/assets/ddc217d3-4de7-4214-bbab-5a71cddf63e4)
![Screenshot 2024-11-09 093948](https://github.com/user-attachments/assets/8a3fb611-c903-41bf-8f11-13d2a32274b0)
![Screenshot 2024-11-09 094011](https://github.com/user-attachments/assets/2f24e63f-1ad8-45cc-b7d3-e11d26f05851)
![Screenshot 2024-11-09 094028](https://github.com/user-attachments/assets/b96b95cc-8667-4459-a02e-cb09ead80eb4)
![Screenshot 2024-11-09 094041](https://github.com/user-attachments/assets/8b70f9c5-0fae-4918-bd9b-80cc14d11b8d)
![Screenshot 2024-11-09 094131](https://github.com/user-attachments/assets/ad54e8cd-04f0-468c-bd2b-6c3cb8b835a1)
![Screenshot 2024-11-09 094202](https://github.com/user-attachments/assets/e05e73af-40c1-43a9-887f-3f9096948d62)





