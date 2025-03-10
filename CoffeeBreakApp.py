import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("API Key not found. Please set GOOGLE_API_KEY in your .env file.")
else:
    genai.configure(api_key=API_KEY)

    def generate_itinerary(passenger_name, arrival_flight, arrival_datetime, departure_flight, departure_datetime, num_pax, num_days, destination, preferences, model_name="models/gemini-1.5-pro-latest"):
        prompt = (
            f"Create a detailed travel itinerary for {passenger_name}.\n"
            f"Destination: {destination}\n"
            f"Arrival Flight: {arrival_flight}, Arrival Date & Time: {arrival_datetime}\n"
            f"Departure Flight: {departure_flight}, Departure Date & Time: {departure_datetime}\n"
            f"Number of Passengers: {num_pax}\n"
            f"Number of Days: {num_days}\n"
            f"Preferences: {preferences}\n\n"
            "Generate:\n"
            "- A detailed accommodation schedule with dates and locations.\n"
            "- A detailed activity schedule, broken down by date and time.\n"
            "- Include local experiences and must-visit places.\n"
            "Format the response clearly and concisely."
        )

        model = genai.GenerativeModel(model_name)
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

    def create_pdf(itinerary_text, passenger_name, destination):
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.setFont("Helvetica", 12)
        lines = itinerary_text.split('\n')
        y = 750  # Starting Y position

        p.drawString(100, 800, f"Travel Itinerary for {passenger_name} in {destination}")
        y -= 20

        for line in lines:
            if y < 50:  # Start a new page if we reach the bottom
                p.showPage()
                p.setFont("Helvetica", 12)
                y = 750
            p.drawString(100, y, line)
            y -= 15

        p.save()
        buffer.seek(0)
        return buffer

    st.title("Comprehensive Travel Itinerary Generator ✈️")

    passenger_name = st.text_input("Passenger Name")
    destination = st.text_input("Destination")
    arrival_flight = st.text_input("Arrival Flight Details")
    arrival_datetime = st.text_input("Arrival Date & Time (YYYY-MM-DD HH:MM)")
    departure_flight = st.text_input("Departure Flight Details")
    departure_datetime = st.text_input("Departure Date & Time (YYYY-MM-DD HH:MM)")
    num_pax = st.number_input("Number of Passengers", min_value=1, step=1)
    num_days = st.number_input("Number of Days", min_value=1, step=1)
    preferences = st.text_area("Enter preferences (e.g., adventure, beaches, culture, food, etc.)", value="Adventure, Food, Culture")

    if st.button("Generate Itinerary"):
        if passenger_name and destination and arrival_flight and arrival_datetime and departure_flight and departure_datetime and num_pax and num_days:
            with st.spinner('Generating Itinerary...'):
                itinerary = generate_itinerary(passenger_name, arrival_flight, arrival_datetime, departure_flight, departure_datetime, num_pax, num_days, destination, preferences)
            if itinerary:
                st.subheader(f"Itinerary for {passenger_name} in {destination}")
                st.markdown(itinerary)

                pdf_buffer = create_pdf(itinerary, passenger_name, destination)
                st.download_button(
                    label="Download Itinerary as PDF",
                    data=pdf_buffer,
                    file_name=f"{passenger_name}_itinerary.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Please fill in all the required fields.")
