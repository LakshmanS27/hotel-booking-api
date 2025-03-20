import pandas as pd

# Load dataset
file_path = "hotel_bookings.csv"  # Update with actual file path if needed
df = pd.read_csv(file_path)

# Handle missing values
df['children'].fillna(0, inplace=True)  # Assuming missing values mean zero children
df['country'].fillna('Unknown', inplace=True)  # Fill missing country with 'Unknown'

# Convert date columns to datetime format
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

# Drop the 'company' column since it has too many missing values
df.drop(columns=['company'], inplace=True)

# Save the cleaned dataset
cleaned_file_path = "cleaned_hotel_bookings.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Data preprocessing complete. Cleaned dataset saved to: {cleaned_file_path}")
