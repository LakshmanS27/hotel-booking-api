import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = "cleaned_hotel_bookings.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Convert date column to datetime
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

# ----- 1. Revenue Trends Over Time -----
df['total_revenue'] = df['adr'] * (df['stays_in_week_nights'] + df['stays_in_weekend_nights'])
df['year_month'] = df['reservation_status_date'].dt.to_period('M')
revenue_trend = df.groupby('year_month')['total_revenue'].sum()

# Plot revenue trend
plt.figure(figsize=(12, 6))
revenue_trend.plot(kind='line', marker='o', color='b')
plt.title('Revenue Trends Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# ----- 2. Cancellation Rate -----
total_bookings = len(df)
canceled_bookings = df['is_canceled'].sum()
cancellation_rate = (canceled_bookings / total_bookings) * 100
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

# ----- 3. Geographical Distribution of Bookings -----
country_counts = df['country'].value_counts().head(10)  # Top 10 countries

# Plot country distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=country_counts.index, y=country_counts.values, palette="viridis")
plt.title("Top 10 Booking Countries")
plt.xlabel("Country")
plt.ylabel("Number of Bookings")
plt.xticks(rotation=45)
plt.show()

# ----- 4. Booking Lead Time Distribution -----
plt.figure(figsize=(12, 6))
sns.histplot(df['lead_time'], bins=50, kde=True, color="blue")
plt.title("Booking Lead Time Distribution")
plt.xlabel("Lead Time (Days)")
plt.ylabel("Number of Bookings")
plt.grid(True)
plt.show()
