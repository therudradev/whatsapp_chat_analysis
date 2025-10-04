from urlextract import URLExtract
from wordcloud import WordCloud
from textblob import TextBlob
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()

#  Compatible emoji detection for all versions of the `emoji` library
def extract_emojis(text):
    """
    Extracts emojis from text, compatible with emoji v1.x to v2.x+.
    """
    if hasattr(emoji, "EMOJI_DATA"):  # Newer versions
        return [c for c in text if c in emoji.EMOJI_DATA]
    elif hasattr(emoji, "UNICODE_EMOJI_ENGLISH"):  # Older versions
        return [c for c in text if c in emoji.UNICODE_EMOJI_ENGLISH]
    elif hasattr(emoji, "UNICODE_EMOJI"):  # Very old versions
        return [c for c in text if c in emoji.UNICODE_EMOJI["en"]]
    else:
        return []


# ===================================================
# 📊 Chat Statistics
# ===================================================
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Total messages
    num_messages = df.shape[0]

    # Total words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # Media count
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # Links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


# ===================================================
# 👥 Most Busy Users
# ===================================================
def most_busy_users(df):
    x = df['user'].value_counts().head()
    percent_df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, percent_df


# ===================================================
# ☁️ WordCloud
# ===================================================
def create_wordcloud(selected_user, df):
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


# ===================================================
# 🏷️ Most Common Words
# ===================================================
def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


# ===================================================
# 😀 Emoji Analysis
# ===================================================
def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend(extract_emojis(message))

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    emoji_df.columns = ['emoji', 'count']

    return emoji_df


# ===================================================
# 📅 Timelines
# ===================================================
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(f"{timeline['month'][i]}-{timeline['year'][i]}")

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date').count()['message'].reset_index()


# ===================================================
# 🕓 Activity Maps
# ===================================================
def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap




# 1️⃣ Sentiment Analysis
def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Compute sentiment polarity (-1 to 1)
    df['sentiment'] = df['message'].apply(lambda msg: TextBlob(msg).sentiment.polarity)

    # Categorize sentiment
    df['sentiment_label'] = df['sentiment'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
    )

    sentiment_summary = df['sentiment_label'].value_counts()
    return df, sentiment_summary


# 3️⃣ Most Active Time of Day
def active_time_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    hourly_activity = df['hour'].value_counts().sort_index()
    return hourly_activity
