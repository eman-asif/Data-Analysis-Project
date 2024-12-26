import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Hotel Bookings Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("hotel_bookings.csv")
    return data

data = load_data()

# Styling
st.markdown(
    """
    <style>
        /* Scrollable Tabs */
        .scrollable-tabs {
            overflow-x: auto;
            white-space: nowrap;
            display: flex;
            gap: 1rem;
            margin-bottom: 20px;
        }
        .scrollable-tabs button {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .scrollable-tabs button:hover {
            background-color: #007bff;
            color: white;
        }
        .scrollable-tabs button.active {
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Scrollable tab selector
tab_options = [
    "Overview", "Booking Cancellations", "ADR Analysis", "Monthly Trends", 
    "Guest Country Analysis", "Stay Length Analysis", "High ADR Analysis",
    "Lead Time Analysis", "Hotel Type Insights", "Special Requests",
    "Weekend vs Weekday Stays", "Market Segment Insights",
    "Deposit Type Analysis", "Customer Type Insights"
]

selected_tab = st.radio(
    label="Navigate Analysis:",
    options=tab_options,
    horizontal=True,
    key="tab_selector"
)

# Tab Analysis
if selected_tab == "Overview":
    st.header("📊 Dataset Overview")
    st.write(data.head(10))

elif selected_tab == "Booking Cancellations":
    st.header("❌ Booking Cancellations")
    cancellation_rate = data['is_canceled'].value_counts(normalize=True) * 100
    st.write(f"Cancellation Rate: {round(cancellation_rate[1], 2)}%")
    fig = px.pie(
        values=cancellation_rate.values,
        names=["Not Canceled", "Canceled"],
        title="Booking Cancellation Distribution",
        color_discrete_sequence=px.colors.qualitative.G10
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "ADR Analysis":
    st.header("💰 Average Daily Rate (ADR) Analysis")
    fig = px.box(data, x='hotel', y='adr', color='hotel', title="ADR Distribution by Hotel")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Monthly Trends":
    st.header("📅 Monthly Booking Trends")
    data['arrival_date'] = pd.to_datetime(data['arrival_date_year'].astype(str) + "-" + data['arrival_date_month'])
    monthly_trends = data.groupby('arrival_date')['hotel'].count().reset_index()
    fig = px.line(monthly_trends, x='arrival_date', y='hotel', title="Monthly Booking Trends")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Guest Country Analysis":
    st.header("🌍 Guest Country Analysis")
    country_data = data['country'].value_counts().reset_index().head(10)
    country_data.columns = ['Country', 'Bookings']
    fig = px.bar(country_data, x='Country', y='Bookings', title="Top 10 Guest Countries")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Stay Length Analysis":
    st.header("🏨 Length of Stay Analysis")
    data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
    fig = px.histogram(data, x='total_nights', title="Length of Stay Distribution")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "High ADR Analysis":
    st.header("📈 High ADR Analysis")
    high_adr_data = data[data['adr'] > data['adr'].mean()]
    fig = px.box(high_adr_data, x='hotel', y='adr', color='hotel', title="ADR Distribution (High ADR)")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Lead Time Analysis":
    st.header("⏳ Lead Time Analysis")
    fig = px.histogram(data, x='lead_time', nbins=50, title="Lead Time Distribution")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Hotel Type Insights":
    st.header("🏨 Hotel Type Insights")
    hotel_data = data['hotel'].value_counts().reset_index()
    hotel_data.columns = ['Hotel Type', 'Bookings']
    fig = px.pie(hotel_data, values='Bookings', names='Hotel Type', title="Hotel Preferences")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Special Requests":
    st.header("🎯 Special Requests Analysis")
    special_request_counts = data['total_of_special_requests'].value_counts()
    fig = px.bar(special_request_counts, x=special_request_counts.index, y=special_request_counts.values, title="Special Requests Count")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Weekend vs Weekday Stays":
    st.header("📆 Weekend vs Weekday Stays")
    data['total_weekday_nights'] = data['stays_in_week_nights']
    data['total_weekend_nights'] = data['stays_in_weekend_nights']
    weekend_vs_weekday = data[['total_weekend_nights', 'total_weekday_nights']].sum()
    fig = px.bar(x=weekend_vs_weekday.index, y=weekend_vs_weekday.values, title="Weekend vs Weekday Stays")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Market Segment Insights":
    st.header("📊 Market Segment Insights")
    market_data = data['market_segment'].value_counts().reset_index()
    market_data.columns = ['Market Segment', 'Bookings']
    fig = px.bar(market_data, x='Market Segment', y='Bookings', title="Market Segment Distribution")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Deposit Type Analysis":
    st.header("💳 Deposit Type Analysis")
    deposit_data = data['deposit_type'].value_counts().reset_index()
    deposit_data.columns = ['Deposit Type', 'Count']
    fig = px.pie(deposit_data, values='Count', names='Deposit Type', title="Deposit Type Distribution")
    st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Customer Type Insights":
    st.header("👤 Customer Type Insights")
    customer_data = data['customer_type'].value_counts().reset_index()
    customer_data.columns = ['Customer Type', 'Count']
    fig = px.bar(customer_data, x='Customer Type', y='Count', title="Customer Type Distribution")
    st.plotly_chart(fig, use_container_width=True)
