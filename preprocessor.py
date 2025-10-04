import re
import pandas as pd

def preprocess(data):
    # Pattern for WhatsApp date-time format like "30/05/25, 12:30 pm - "
    pattern = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s*[apAP][mM]\s*-\s"

    # Split messages and extract dates
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Clean invisible Unicode spaces (non-breaking spaces, narrow no-break spaces)
    df['message_date'] = (
        df['message_date']
        .str.replace('\u202f', ' ', regex=True)
        .str.replace('\u00a0', ' ', regex=True)
    )

    # Convert to datetime safely (supports 2-digit years & 12-hour time)
    df['message_date'] = pd.to_datetime(
        df['message_date'],
        format='%d/%m/%y, %I:%M %p - ',
        errors='coerce'
    )

    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Split user and message text
    users = []
    messages = []

    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if len(entry) > 2:  # message with username
            users.append(entry[1])
            messages.append(" ".join(entry[2:]).strip())
        else:  # group notification (system message)
            users.append('group_notification')
            messages.append(entry[0].strip())

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract useful date-time features
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Create period column (for heatmap)
    period = []
    for hour in df['hour']:
        if pd.isna(hour):
            period.append(None)
        elif hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append(f"00-{hour + 1}")
        else:
            period.append(f"{hour}-{hour + 1}")

    df['period'] = period

    return df
