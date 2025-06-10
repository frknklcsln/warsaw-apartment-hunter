"""
Warsaw Apartment Hunter - Streamlit Dashboard
==============================================

A comprehensive dashboard for finding apartments in Warsaw with optimal commute analysis.
Uses pre-optimized GTFS transport data for fast performance and deployment.

Features:
- Interactive apartment search with commute time analysis
- Real-time map visualization with route information
- Advanced analytics and market insights
- Quick address feasibility checker
- Optimized for Streamlit Community Cloud deployment

Author: Furkan
Date: 2025-06-09
Version: 2.0 (Optimized for deployment)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.analysis import ApartmentAnalyzer
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import time
import re
from collections import defaultdict
from geopy.geocoders import Nominatim
import folium
import os
import subprocess
import sys
import plotly.express as px
import plotly.graph_objects as go
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Warsaw Apartment Hunter",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CONFIGURATION SETUP
# ==========================================

# Get project root directory
project_root = Path(__file__).resolve().parent
# Application configuration - uses optimized/filtered GTFS data (99% smaller than original)
CONFIG = {
    # Apartment data source
    'apartments_path': project_root / 'data' / 'apartment' / 'warsaw_private_owner_apartments.xlsx',
    'apartments_sheet': 'Private_Owner_Apartments',

    # Optimized transport data (filtered to only include routes serving work location)
    # These files are 99% smaller than original GTFS data while maintaining full accuracy
    'stops_path': project_root / 'data' / 'transport' / 'stops.txt',
    'stop_times_path': project_root / 'data' / 'transport' / 'stop_times.txt',
    'trips_path': project_root / 'data' / 'transport' / 'trips.txt',
    'routes_path': project_root / 'data' / 'transport' / 'routes.txt',
    'shapes_path': project_root / 'data' / 'transport' / 'shapes.txt',
    'calendar_path': project_root / 'data' / 'transport' / 'calendar.txt',
    'calendar_dates_path': project_root / 'data' / 'transport' / 'calendar_dates.txt',

    # Work location coordinates (office location)
    'office_lat': 52.182348,
    'office_lon': 20.965572,

    # Search parameters for commute analysis
    'apartment_max_distance': 1000,  # Maximum walking distance from apartment to transit (meters)
    'bus_radius_m': 1000,            # Bus stop search radius from office (meters)
    'tram_radius_m': 1000,            # Tram stop search radius from office (meters)
}

# ==========================================
# STYLING AND CSS
# ==========================================

st.markdown("""
<style>
    /* Main container styling for better layout */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Header styling with gradient background */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 9px;
        text-align: center;
        margin-top: -5rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 4px 4px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Chart container styling with dark theme for analytics */
    .chart-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* KPI card styling for metrics display */
    .kpi-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    /* Metric cards styling with hover effects */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    /* Filter container styling for sidebar */
    .filter-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    
    /* Info box styling for notifications */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border: 1px solid #bbdefb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #1565c0;
        font-weight: 500;
    }
    
    /* Apartment card styling for list view */
    .apartment-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .apartment-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        transform: translateY(-1px);
    }
    
    /* Special styling for new listings */
    .apartment-card.new-today {
        border-left: 4px solid #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
    }
    
    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def geocode_address_robust(address):
    """
    Robust geocoding with multiple strategies for Warsaw addresses.

    This function attempts to geocode Polish addresses using various strategies:
    1. Normalize address by removing Polish street prefixes (ul., al., pl.)
    2. Add Warsaw context if not present
    3. Try multiple variations of the address
    4. Validate coordinates are within Warsaw bounds

    Args:
        address (str): Address to geocode

    Returns:
        tuple: (latitude, longitude) or None if not found
    """
    try:
        # Normalize the address - remove common Polish street prefixes
        normalized_address = address
        polish_prefixes = ["ul. ", "ul.", "ulica ", "Ulica ", "al. ", "al.", "aleja ", "Aleja ", "pl. ", "pl.", "plac ", "Plac "]

        for prefix in polish_prefixes:
            if normalized_address.lower().startswith(prefix.lower()):
                normalized_address = normalized_address[len(prefix):].strip()
                break

        # Add Warsaw to address if not present
        if "warsaw" not in normalized_address.lower() and "warszawa" not in normalized_address.lower():
            normalized_address += ", Warsaw, Poland"

        geolocator = Nominatim(user_agent="warsaw_apartment_hunter_v3")

        # Try the normalized address first (without ul., etc.)
        location = geolocator.geocode(normalized_address, timeout=15)

        if location:
            lat, lon = location.latitude, location.longitude
            # Check if within Warsaw bounds (safety check)
            if 51.9 <= lat <= 52.5 and 20.6 <= lon <= 21.4:
                return (lat, lon)

        # If that didn't work, try the original address
        if "warsaw" not in address.lower() and "warszawa" not in address.lower():
            address_with_city = address + ", Warsaw, Poland"
        else:
            address_with_city = address

        location = geolocator.geocode(address_with_city, timeout=15)

        if location:
            lat, lon = location.latitude, location.longitude
            if 51.9 <= lat <= 52.5 and 20.6 <= lon <= 21.4:
                return (lat, lon)

        return None
    except Exception as e:
        print(f"Geocoding error: {str(e)}")
        return None

def find_best_route_from_location(lat, lon):
    """
    Find the best route from a specific location to work using optimized transport data.

    This function uses the pre-filtered GTFS data to calculate optimal routes:
    1. Find nearby transit stops within walking distance
    2. Calculate routes using only work-relevant trips and stops
    3. Consider walking time + transit time + final walking time
    4. Return the route with minimum total travel time

    Args:
        lat (float): Latitude of starting location
        lon (float): Longitude of starting location

    Returns:
        dict: Route information with travel times and details, or None if no route found
    """
    try:
        # Find nearby transit stops within walking distance
        nearby_stops = analyzer._find_nearby_stops(lat, lon, CONFIG['apartment_max_distance'])

        if not nearby_stops:
            return None

        best_route = None
        min_total_time = float('inf')

        # Get work arrivals within time window (8:00-9:00 AM)
        work_arrivals = analyzer.stop_times[
            (analyzer.stop_times['stop_id'].isin(analyzer.work_stop_ids)) &
            (analyzer.stop_times['arrival_time'].dt.time >= analyzer.work_arrival_time_early) &
            (analyzer.stop_times['arrival_time'].dt.time <= analyzer.work_arrival_time_late) &
            (analyzer.stop_times['arrival_time'].notna())
        ]

        valid_trip_ids = set(work_arrivals['trip_id'])

        if not valid_trip_ids:
            return None

        # Create lookup dictionaries for performance optimization
        route_metadata = dict(
            zip(analyzer.routes['route_id'],
                zip(analyzer.routes['route_short_name'], analyzer.routes['route_type']))
        )
        stop_coords = dict(
            zip(analyzer.stops['stop_id'],
                zip(analyzer.stops['stop_lat'], analyzer.stops['stop_lon']))
        )
        trip_lookup = dict(
            zip(analyzer.trips['trip_id'],
                zip(analyzer.trips['route_id'], analyzer.trips['shape_id']))
        )

        # Process stop times for valid trips
        valid_stop_times = analyzer.stop_times[analyzer.stop_times['trip_id'].isin(valid_trip_ids)][
            ['trip_id', 'stop_id', 'departure_time', 'arrival_time']
        ].copy()

        trip_stop_times = {
            trip_id: dict(zip(group['stop_id'], zip(group['departure_time'], group['arrival_time'])))
            for trip_id, group in valid_stop_times.groupby('trip_id')
        }

        # Find work stops for each route
        route_work_stops = {
            (route_id, shape_id): [ws for ws in analyzer.work_stop_ids if ws in stops_seq]
            for (route_id, shape_id), stops_seq in analyzer.shape_stop_order.items()
        }

        route_valid_trips = defaultdict(list)
        for trip_id in valid_trip_ids:
            if trip_id in trip_lookup:
                route_valid_trips[trip_lookup[trip_id]].append(trip_id)

        # Find best route from each nearby stop
        for stop_id, walking_dist_from_apt in nearby_stops:
            routes_for_stop = analyzer.stop_to_routes.get(stop_id, [])

            for route_id, shape_id in routes_for_stop:
                stops_seq = analyzer.shape_stop_order.get((route_id, shape_id), [])
                if not stops_seq:
                    continue

                try:
                    stop_idx = stops_seq.index(stop_id)
                except ValueError:
                    continue

                valid_work_stops = [
                    ws for ws in route_work_stops.get((route_id, shape_id), [])
                    if ws in stops_seq and stops_seq.index(ws) > stop_idx
                ]

                if not valid_work_stops:
                    continue

                # Calculate travel times for each work stop
                for work_stop in valid_work_stops:
                    for trip_id in route_valid_trips.get((route_id, shape_id), []):
                        trip_data = trip_stop_times.get(trip_id, {})

                        if stop_id in trip_data and work_stop in trip_data:
                            departure, arrival = trip_data[stop_id][0], trip_data[work_stop][1]

                            transit_time_mins = (arrival - departure).total_seconds() / 60
                            walking_time_from_apt_mins = walking_dist_from_apt / 83.33  # 5 km/h walking speed

                            work_stop_coords = stop_coords.get(work_stop)
                            if not work_stop_coords:
                                continue

                            # Calculate walking distance from work stop to office
                            from src.analysis import haversine_np
                            walking_dist_to_office = haversine_np(
                                work_stop_coords[0], work_stop_coords[1],
                                CONFIG['office_lat'], CONFIG['office_lon']
                            )
                            walking_time_to_office_mins = walking_dist_to_office / 83.33

                            total_time = walking_time_from_apt_mins + transit_time_mins + walking_time_to_office_mins

                            # Update best route if this is better
                            if 0 < transit_time_mins < 120 and total_time < min_total_time:
                                min_total_time = total_time
                                route_short_name, route_type_code = route_metadata.get(route_id, ('Unknown', 3))

                                best_route = {
                                    'departure_time': departure.strftime('%H:%M'),
                                    'arrival_time': arrival.strftime('%H:%M'),
                                    'transit_time': transit_time_mins,
                                    'walking_time_from_apt': walking_time_from_apt_mins,
                                    'walking_time_to_office': walking_time_to_office_mins,
                                    'total_time': total_time,
                                    'route_short_name': route_short_name,
                                    'route_type': 'Bus' if route_type_code == 3 else 'Tram',
                                    'boarding_stop': stop_id,
                                    'boarding_stop_name': analyzer.stops[analyzer.stops['stop_id'] == stop_id]['stop_name'].iloc[0] if not analyzer.stops[analyzer.stops['stop_id'] == stop_id].empty else 'Unknown',
                                    'destination_stop': work_stop,
                                    'destination_stop_name': analyzer.stops[analyzer.stops['stop_id'] == work_stop]['stop_name'].iloc[0] if not analyzer.stops[analyzer.stops['stop_id'] == work_stop].empty else 'Unknown',
                                    'route_id': route_id,
                                    'shape_id': shape_id,
                                    'walking_distance': walking_dist_from_apt
                                }

        return best_route

    except Exception as e:
        print(f"Error finding route: {str(e)}")
        return None

def check_address_feasibility(address, max_travel_time=45):
    """
    Check if an address would be feasible for commuting using optimized transport data.

    This function provides quick feasibility analysis for any Warsaw address:
    1. Geocode the address to get coordinates
    2. Find nearby transit stops
    3. Calculate best route to work using optimized data
    4. Return feasibility assessment with detailed information

    Args:
        address (str): Address to check
        max_travel_time (int): Maximum acceptable travel time in minutes

    Returns:
        dict: Feasibility analysis results with route information
    """
    try:
        # Geocode the address
        coords = geocode_address_robust(address)

        if not coords:
            return {
                "error": f"Could not find '{address}' in Warsaw area. Try a more specific address."
            }

        lat, lon = coords
        nearby_stops = analyzer._find_nearby_stops(lat, lon, CONFIG['apartment_max_distance'])

        if not nearby_stops:
            return {
                "feasible": False,
                "reason": f"No transit stops within {CONFIG['apartment_max_distance']}m walking distance",
                "address": address,
                "coordinates": (lat, lon),
                "nearby_stops": 0
            }

        # Find best route from this location
        best_route = find_best_route_from_location(lat, lon)

        if best_route and best_route['total_time'] <= max_travel_time:
            return {
                "feasible": True,
                "travel_time": best_route['total_time'],
                "route_info": best_route,
                "address": address,
                "coordinates": (lat, lon),
                "nearby_stops": len(nearby_stops)
            }
        elif best_route:
            return {
                "feasible": False,
                "reason": f"Travel time too long: {best_route['total_time']:.1f} min (max: {max_travel_time} min)",
                "address": address,
                "coordinates": (lat, lon),
                "nearby_stops": len(nearby_stops),
                "route_info": best_route
            }
        else:
            work_stops_count = len(analyzer.work_stop_ids)
            direct_routes_count = len(analyzer.direct_route_ids)

            return {
                "feasible": False,
                "reason": f"No direct routes found from {len(nearby_stops)} nearby stops to work area ({work_stops_count} work stops, {direct_routes_count} direct routes in system)",
                "address": address,
                "coordinates": (lat, lon),
                "nearby_stops": len(nearby_stops)
            }

    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}."
        }

def display_quick_result(result):
    """
    Display the feasibility check result in the sidebar with detailed breakdown.

    Args:
        result (dict): Result from check_address_feasibility
    """
    if "error" in result:
        st.sidebar.error(f"❌ {result['error']}")
        with st.sidebar.expander("💡 Try these formats"):
            st.write("• ul. Marszałkowska 1")
            st.write("• Plac Konstytucji 4")
            st.write("• Mokotów")
            st.write("• Żoliborz")
            st.write("• al. Jerozolimskie 100")
        return

    address = result['address']

    if result['feasible']:
        travel_time = result['travel_time']
        route_info = result['route_info']

        st.sidebar.success(f"✅ **FEASIBLE!**")
        st.sidebar.markdown(f"📍 **{address}**")
        st.sidebar.markdown(f"🕐 **{travel_time:.1f} minutes** to work")
        st.sidebar.markdown(f"🚌 Via **{route_info['route_short_name']}** ({route_info['route_type']})")
        st.sidebar.markdown(f"🚏 **{result['nearby_stops']}** nearby stops")

        with st.sidebar.expander("📊 Travel Time Breakdown"):
            st.write(f"🚶 Walk to stop: **{route_info['walking_time_from_apt']:.1f} min**")
            st.write(f"   → {route_info.get('boarding_stop_name', 'Stop')} ({route_info['walking_distance']:.0f}m)")
            st.write(f"🚌 Transit time: **{route_info['transit_time']:.1f} min**")
            st.write(f"   → Route {route_info['route_short_name']} ({route_info['route_type']})")
            st.write(f"🚶 Walk to office: **{route_info['walking_time_to_office']:.1f} min**")
            st.write(f"   → From {route_info.get('destination_stop_name', 'Stop')}")
            st.write(f"🕐 Departure: **{route_info['departure_time']}**")
            st.write(f"🏢 Arrival: **{route_info['arrival_time']}**")

        if st.sidebar.button("📍 Show on Map", key="show_on_map_btn"):
            st.session_state['temp_location'] = result
            st.session_state['map_needs_update'] = True
            st.rerun()

    else:
        st.sidebar.error(f"❌ **NOT FEASIBLE**")
        st.sidebar.markdown(f"📍 **{address}**")
        st.sidebar.markdown(f"**Reason:** {result['reason']}")

        if result.get('nearby_stops', 0) > 0:
            st.sidebar.markdown(f"🚏 {result['nearby_stops']} stops nearby")

            if 'route_info' in result:
                route_info = result['route_info']
                with st.sidebar.expander("📊 Available Route (too slow)"):
                    st.write(f"🚌 Route: **{route_info['route_short_name']}** ({route_info['route_type']})")
                    st.write(f"🕐 Total time: **{route_info['total_time']:.1f} min**")
                    st.write(f"   • Walk to stop: {route_info['walking_time_from_apt']:.1f} min")
                    st.write(f"   • Transit: {route_info['transit_time']:.1f} min")
                    st.write(f"   • Walk to office: {route_info['walking_time_to_office']:.1f} min")

def get_location_display(apartment):
    """
    Get proper location display for apartment from available fields.

    Tries multiple location fields in order of preference to find the best
    available location description for display purposes.

    Args:
        apartment (pd.Series): Apartment data

    Returns:
        str: Best available location description
    """
    # Try multiple location fields in order of preference
    location_fields = [
        'extracted_street',
        'raw_address',
        'extracted_district',
        'district',
        'address',
        'location'
    ]

    for field in location_fields:
        if field in apartment and pd.notna(apartment[field]) and str(apartment[field]).lower() != 'nan':
            location = str(apartment[field]).strip()
            if location and location.lower() != 'nan':
                # Clean up the location string
                if location.startswith('ul. '):
                    return location
                elif ',' in location:
                    # Take the first part before comma if it looks like a street
                    parts = location.split(',')
                    if len(parts[0].strip()) > 3:
                        return parts[0].strip()
                return location

    # If no location found, try to extract from title or other fields
    if 'title' in apartment and pd.notna(apartment['title']):
        title = str(apartment['title'])
        # Try to extract street name from title
        street_pattern = r'(ul\.|al\.|pl\.)\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s]+)'
        match = re.search(street_pattern, title, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {match.group(2).strip()}"

    return "Location not specified"

def run_data_refresh_script(script_name, script_description):
    """
    Run a data refresh script and handle the process with proper error handling.

    This function executes Python scripts for data updates (like scraper.py)
    and provides user feedback through the Streamlit interface.

    Args:
        script_name (str): Name of the script to run
        script_description (str): Description for user feedback

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        script_path = project_root / script_name

        if not script_path.exists():
            st.error(f"❌ Script {script_name} not found at {script_path}")
            return False

        with st.spinner(f"🔄 Running {script_description}..."):
            # Run the script using the same Python executable
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                st.success(f"✅ {script_description} completed successfully!")

                # Show last few lines of output if available
                if result.stdout:
                    output_lines = result.stdout.strip().split('\n')
                    if len(output_lines) > 0:
                        with st.expander(f"📋 {script_description} Output"):
                            # Show last 10 lines
                            for line in output_lines[-10:]:
                                st.text(line)

                return True
            else:
                st.error(f"❌ {script_description} failed with return code {result.returncode}")

                if result.stderr:
                    with st.expander("🔍 Error Details"):
                        st.code(result.stderr, language="text")

                return False

    except subprocess.TimeoutExpired:
        st.error(f"❌ {script_description} timed out after 5 minutes")
        return False
    except Exception as e:
        st.error(f"❌ Error running {script_description}: {str(e)}")
        return False

def clear_all_caches_and_reset():
    """
    Clear all caches and reset session state for fresh data loading.

    This function clears Streamlit caches and resets application state
    to force fresh data loading from Excel and GTFS files.
    """
    # Clear Streamlit caches
    st.cache_resource.clear()
    st.cache_data.clear()

    # Reset Excel modification time to force reload
    if 'excel_mod_time' in st.session_state:
        del st.session_state.excel_mod_time

    # Reset all application state
    st.session_state.filter_applied = False
    st.session_state.apartment_routes = {}
    st.session_state.filtered_df = pd.DataFrame()
    st.session_state.needs_route_calculation = True
    st.session_state.map_data = None
    st.session_state.map_needs_update = True

    # Clear temporary location
    if 'temp_location' in st.session_state:
        st.session_state.temp_location = None


# ==========================================
# CHART CREATION FUNCTIONS (AESTHETIC & BUG-FIX UPDATE)
# ==========================================

# Define a modern, pastel color palette for the charts, perfect for a dark background
PASTEL_GRADIENT = {
    'price_hist': ['#a1c4fd', '#c2e9fb'],  # Sky blue gradient
    'travel_hist': ['#d4b2f5', '#9a67ea'],  # Peach-pink gradient
    'timeline_fill': 'rgba(168, 237, 234, 0.3)',  # Soft mint-green fill
    'timeline_line': '#a8edea',  # Soft mint-green line
    'scatter_discrete': px.colors.qualitative.Pastel,  # A beautiful pastel set for rooms
}


def create_price_distribution_chart(df):
    """Creates a modern, styled histogram for price distribution with a pastel gradient."""
    if df.empty or 'price' not in df.columns: return go.Figure()

    fig = px.histogram(df, x="price", nbins=30, title="💰 Price Distribution",
                       color_discrete_sequence=PASTEL_GRADIENT['price_hist'])
    fig.update_layout(
        title_font_size=22, title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#EAEAEA',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Price (PLN)"),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Number of Apartments"),
        bargap=0.1, legend_title_text=''
    )
    fig.update_traces(marker_line=dict(width=1.5, color='#1a1a2e'))
    return fig


def create_travel_time_distribution_chart(df, apartment_routes):
    """Creates a modern, styled histogram for travel time distribution. (BUG FIXED)"""
    if df.empty or not apartment_routes or 'listing_id' not in df.columns: return go.Figure()

    # BUG FIX: Ensure we only use travel times for apartments in the *final* filtered list.
    valid_listing_ids = set(df['listing_id'])
    travel_times = [
        route.get('total_time', 0)
        for listing_id, route in apartment_routes.items()
        if listing_id in valid_listing_ids
    ]
    if not travel_times: return go.Figure()

    fig = px.histogram(x=travel_times, nbins=20, title="🚌 Travel Time Distribution",
                       color_discrete_sequence=PASTEL_GRADIENT['travel_hist'])
    fig.update_layout(
        title_font_size=22, title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#EAEAEA',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Travel Time (minutes)"),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Number of Apartments"),
        bargap=0.1, legend_title_text=''
    )
    fig.update_traces(marker_line=dict(width=1.5, color='#1a1a2e'))
    return fig


def create_price_vs_commute_scatter(df, apartment_routes):
    """Creates a modern scatter plot of Price vs. Travel Time. (BUG FIXED)"""
    if df.empty or not apartment_routes or 'listing_id' not in df.columns: return go.Figure()

    scatter_data = []
    # BUG FIX: Use the 'listing_id' column to reliably look up the correct route.
    for _, apartment in df.iterrows():
        listing_id = apartment['listing_id']
        route = apartment_routes.get(listing_id)
        if route:
            scatter_data.append({
                'travel_time': route.get('total_time', 0),
                'price': apartment['price'],
                'area': apartment.get('area', 50),
                'rooms': f"{int(apartment.get('rooms', 1))} rooms"
            })
    if not scatter_data: return go.Figure()

    scatter_df = pd.DataFrame(scatter_data)
    fig = px.scatter(
        scatter_df, x="travel_time", y="price", size="area", color="rooms",
        title="💸 Price vs. Commute Time (Size by Area, Color by Rooms)",
        size_max=20,
        color_discrete_sequence=PASTEL_GRADIENT['scatter_discrete'],
        hover_data={'price': ':,', 'travel_time': ':.1f', 'area': True}
    )
    fig.update_layout(
        title_font_size=22, title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#EAEAEA',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Travel Time (minutes)"),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Price (PLN)"),
        legend_title_text='Rooms'
    )
    return fig


def create_listings_timeline_chart(df):
    """Creates a modern, gradient area chart showing new listings over time."""
    if df.empty or 'created_at_date' not in df.columns: return go.Figure()

    date_counts = df['created_at_date'].value_counts().sort_index()
    fig = px.area(date_counts, x=date_counts.index, y=date_counts.values, title="📅 New Listings Timeline")
    fig.update_layout(
        title_font_size=22, title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#EAEAEA',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Date"),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Number of New Listings"),
        showlegend=False
    )
    fig.update_traces(
        line=dict(color=PASTEL_GRADIENT['timeline_line'], width=2.5),
        fillcolor=PASTEL_GRADIENT['timeline_fill'],
        fill='tozeroy'
    )
    return fig


# ==========================================
# MAIN APP INITIALIZATION
# ==========================================

# Load analyzer on first run using optimized transport data
# This will create cache files with the optimized (99% smaller) data if they don't exist
if 'analyzer' not in st.session_state:
    with st.spinner("🔄 Loading Warsaw Apartment Hunter with optimized transport data..."):
        try:
            # Initialize analyzer with optimized GTFS data
            st.session_state.analyzer = ApartmentAnalyzer(CONFIG)
            st.session_state.excel_mod_time = os.path.getmtime(CONFIG['apartments_path']) if os.path.exists(CONFIG['apartments_path']) else 0
        except Exception as e:
            st.error(f"❌ Error during initialization: {str(e)}")
            st.stop()

        # Reset related session state
        st.session_state.filter_applied = False
        st.session_state.apartment_routes = {}
        st.session_state.filtered_df = pd.DataFrame()
        st.session_state.needs_route_calculation = True
        st.session_state.map_data = None
        st.session_state.map_needs_update = True

# Get analyzer instance
analyzer = st.session_state.analyzer

# Initialize session state variables for application state management
if 'map_data' not in st.session_state:
    st.session_state.map_data = None
if 'map_needs_update' not in st.session_state:
    st.session_state.map_needs_update = True
if 'filter_applied' not in st.session_state:
    st.session_state.filter_applied = False
if 'current_filters' not in st.session_state:
    # Set default filters based on available data
    min_date, max_date = analyzer.get_date_range_for_apartments()
    default_start_date = max_date - timedelta(days=7)

    st.session_state.current_filters = {
        'selected_rooms': [2, 3],
        'max_price': 4000,
        'max_travel_time': 45,
        'date_range': (default_start_date, max_date)
    }
if 'apartment_routes' not in st.session_state:
    st.session_state.apartment_routes = {}
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame()
if 'needs_route_calculation' not in st.session_state:
    st.session_state.needs_route_calculation = True
if 'temp_location' not in st.session_state:
    st.session_state.temp_location = None

# ==========================================
# MAIN APP UI
# ==========================================

# Main header with information about optimized data
st.markdown("""
<div class="main-header">
    <h1>🏠 Warsaw Apartment Hunter</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">An Intelligent Multi-Objective Optimization System for Urban Housing Selection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with filters and tools
with st.sidebar:
    st.markdown("## 🔍 Search Filters")
    st.markdown("---")
    # Property Requirements Section
    st.markdown('<div style="color: #00000; font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #667eea;">🏠 Property Requirements</div>', unsafe_allow_html=True)

    with st.form(key='filter_form'):
        selected_rooms = st.multiselect(
            "Number of Rooms",
            options=sorted(analyzer.apartments['rooms'].unique()),
            default=st.session_state.current_filters['selected_rooms'],
            help="Select the number of rooms you're looking for"
        )

        max_price = st.slider(
            "Maximum Price (PLN)",
            min_value=1500,
            max_value=10000,
            value=st.session_state.current_filters['max_price'],
            step=100,
            help="Set your maximum budget"
        )

        max_travel_time = st.slider(
            "Max Travel Time (minutes)",
            15, 90,
            st.session_state.current_filters['max_travel_time'],
            5,
            help="Maximum acceptable commute time to work"
        )

        min_date, max_date = analyzer.get_date_range_for_apartments()

        date_range = st.slider(
            "Date Range",
            min_value=min_date,
            max_value=max_date,
            value=st.session_state.current_filters['date_range'],
            format="YYYY-MM-DD",
            help="Filter apartments by their creation date"
        )

        submitted = st.form_submit_button(
            "🔍 Find Apartments",
            type="primary",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Address checker section
    st.markdown("---")
    st.markdown("## 🎯 Quick Address Check")

    address_input = st.text_input(
        "Enter street address:",
        placeholder="ul. Marszałkowska 1",
        help="Check if a specific address is feasible for commuting"
    )

    col1, col2 = st.columns(2)
    with col1:
        max_travel_time_check = st.slider(
            "Max time (min)",
            15, 90, 45, 5,
            key="address_check_time"
        )
    with col2:
        show_on_map = st.checkbox(
            "Show on map",
            value=True,
            key="show_address_on_map"
        )

    check_clicked = st.button(
        "🔍 Check Feasibility",
        type="primary",
        key="check_feasibility_btn",
        use_container_width=True
    )

    if check_clicked:
        if address_input.strip():
            with st.spinner("Checking address..."):
                result = check_address_feasibility(address_input.strip(), max_travel_time_check)
            display_quick_result(result)
            if show_on_map and 'coordinates' in result:
                st.session_state.temp_location = result
                st.session_state.map_needs_update = True
        else:
            st.warning("Please enter an address")

    if st.button("🗑️ Clear", key="clear_result_btn", use_container_width=True):
        st.session_state.temp_location = None
        st.session_state.map_needs_update = True
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Data refresh section (TRANSPORT UPDATE BUTTON REMOVED AS REQUESTED)
    st.markdown("---")
    st.markdown("## 🔄 Data Management")
    st.markdown("Manage apartment data and refresh the dashboard:")

    # Apartment Data Refresh (kept as requested)
    # --- THIS SECTION REPLACES THE MANUAL UPDATE BUTTON ---

    # Display the last update time of the apartment data file
    st.markdown("**🏠 Apartment Data Status**")
    try:
        # Use the file path from your CONFIG at the top of the file
        data_file_path = CONFIG['apartments_path']

        if os.path.exists(data_file_path):
            # Get the file's last modification time as a timestamp
            last_modified_timestamp = os.path.getmtime(data_file_path)
            # Convert the timestamp to a readable datetime object
            last_updated_time = datetime.fromtimestamp(last_modified_timestamp)

            # Display the last update time in a user-friendly format
            st.success(f"Data last updated: {last_updated_time.strftime('%Y-%m-%d %H:%M')} UTC")
            st.info("Data is automatically refreshed multiple times a day via GitHub Actions.")
        else:
            st.error("Apartment data file not found. Please run the initial scraper.")

    except Exception as e:
        st.error(f"Could not read data update time: {str(e)}")

    # --- THE MANUAL CACHE CLEAR BUTTON REMAINS THE SAME ---

    # Manual Cache Clear
    st.markdown("**🗑️ Cache Management**")
    if st.button(
        "🗑️ Clear Cache",
        key="clear_cache_btn",
        use_container_width=True,
        help="Clear all caches and refresh dashboard"
    ):
        clear_all_caches_and_reset()
        st.success("🗑️ Cache cleared and dashboard refreshed!")
        time.sleep(1)
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# FILTER PROCESSING
# ==========================================

# Process filter form submission
if submitted:
    new_filters = {
        'selected_rooms': selected_rooms,
        'max_price': max_price,
        'max_travel_time': max_travel_time,
        'date_range': date_range
    }

    # Check if route-affecting filters changed
    filters_that_affect_routes = ['selected_rooms', 'max_price', 'max_travel_time']
    route_affecting_filters_changed = any(
        st.session_state.current_filters.get(key) != new_filters[key]
        for key in filters_that_affect_routes
    )

    if route_affecting_filters_changed:
        st.session_state.needs_route_calculation = True
        st.session_state.map_needs_update = True

    st.session_state.current_filters = new_filters
    st.session_state.filter_applied = True

    # ENHANCED: Check for feasible solutions and provide feedback
    if st.session_state.filter_applied and st.session_state.needs_route_calculation:
        with st.spinner("🔄 Checking apartment availability..."):
            # Quick feasibility check
            apartments_df = analyzer.apartments

            # Step-by-step filtering with counts
            original_count = len(apartments_df)

            # Room filter
            after_rooms = apartments_df[apartments_df['rooms'].isin(selected_rooms)]
            rooms_count = len(after_rooms)

            # Price filter
            after_price = after_rooms[after_rooms['price'] <= max_price]
            price_count = len(after_price)

            # Basic feasibility check (before full route calculation)
            if rooms_count == 0:
                st.warning("🚫 **No apartments match your criteria**")
                st.info("💡 **Try**: Selecting different room counts")
            elif price_count == 0:
                st.warning("🚫 **No apartments match your criteria**")
                st.info(f"💡 **Try**: Increasing max price above {max_price:,} PLN")
            elif price_count < 5:
                st.warning(f"⚠️ **Only {price_count} apartments** match your price and room criteria")
                st.info("💡 **Consider**: Slightly increasing your budget or expanding room options")

# ==========================================
# ROUTE CALCULATION
# ==========================================

# Calculate routes if needed with progress indication
if st.session_state.filter_applied and st.session_state.needs_route_calculation:
    filters = st.session_state.current_filters

    with st.spinner("🔄 Finding routes and preparing data using optimized transport data..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🏠 Loading apartment data...")
        progress_bar.progress(20)

        # Run analysis using optimized transport data
        apartment_routes, new_today_listings = analyzer.run_interactive_analysis(
            filters['max_price'], filters['selected_rooms']
        )

        status_text.text("🚌 Calculating routes with optimized data...")
        progress_bar.progress(60)

        st.session_state.apartment_routes = apartment_routes
        st.session_state.new_today_listings = new_today_listings

        filtered_df = analyzer.filtered_apartments.copy()

        status_text.text("📊 Processing results...")
        progress_bar.progress(80)

        # Ensure listing_id column exists for proper data handling
        if 'listing_id' not in filtered_df.columns:
            if filtered_df.index.name != 'listing_id':
                filtered_df = filtered_df.reset_index()
                if 'index' in filtered_df.columns:
                    filtered_df = filtered_df.rename(columns={'index': 'listing_id'})
                elif 'listing_id' not in filtered_df.columns:
                    filtered_df['listing_id'] = range(len(filtered_df))

        # Process date information for new listing detection
        if 'created_at' in filtered_df.columns:
            filtered_df['created_at'] = pd.to_datetime(filtered_df['created_at'], errors='coerce')
            filtered_df['created_at_date'] = filtered_df['created_at'].dt.date
            today = datetime.today().date()
            filtered_df['days_since_created'] = (pd.Timestamp(today) - filtered_df['created_at']).dt.days
            filtered_df['is_new_today'] = filtered_df['listing_id'].isin(new_today_listings)
        else:
            filtered_df['days_since_created'] = 0
            filtered_df['is_new_today'] = False

        status_text.text("✅ Finalizing...")
        progress_bar.progress(100)

        st.session_state.filtered_df = filtered_df
        st.session_state.needs_route_calculation = False

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

# ==========================================
# RESULTS DISPLAY / WELCOME MESSAGE
# ==========================================

# This single, clean if/else block correctly handles all three states:
# 1. Filters applied with results
# 2. Filters applied with NO results
# 3. No filters applied yet (initial load / refresh)

if st.session_state.get('filter_applied', False):
    # This block runs ONLY after the "Find Apartments" button has been clicked.
    # It will never show the welcome message.

    # Start with the dataframe that has been pre-filtered by price and rooms
    df = st.session_state.filtered_df.copy()
    apartment_routes = st.session_state.apartment_routes
    filters = st.session_state.current_filters
    new_today_listings = st.session_state.get('new_today_listings', set())

    # Apply the final travel time and date filters to get the definitive list
    if apartment_routes:
        valid_apartments = [route['listing_id'] for route in apartment_routes.values()
                            if route.get('total_time', float('inf')) <= filters['max_travel_time']]
        if 'listing_id' in df.columns:
            df = df[df['listing_id'].isin(valid_apartments)]
        else:
            df = df[df.index.isin(valid_apartments)]

    if 'created_at_date' in df.columns:
        start_date, end_date = filters['date_range']
        df = df[
            (df['created_at_date'] >= start_date) &
            (df['created_at_date'] <= end_date)
            ]

    final_filtered_df = df

    if not final_filtered_df.empty:
        # STATE 1: RESULTS FOUND - Display all metrics, maps, and lists.

        st.markdown("## 📊 Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_count = len(final_filtered_df)
            st.markdown(
                f"""<div class="metric-card"><div style="font-size: 0.9rem; color: #718096; font-weight: 500; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">Total Apartments</div><div style="font-size: 2rem; font-weight: 700; color: #2d3748; margin: 0.5rem 0;">{total_count}</div></div>""",
                unsafe_allow_html=True)
        with col2:
            today_count = len(final_filtered_df[final_filtered_df.get('is_new_today', False) == True])
            st.markdown(
                f"""<div class="metric-card"><div style="font-size: 0.9rem; color: #718096; font-weight: 500; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">New Today</div><div style="font-size: 2rem; font-weight: 700; color: #f59e0b; margin: 0.5rem 0;">{today_count}</div></div>""",
                unsafe_allow_html=True)
        with col3:
            avg_price = int(final_filtered_df['price'].mean())
            st.markdown(
                f"""<div class="metric-card"><div style="font-size: 0.9rem; color: #718096; font-weight: 500; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">Average Price</div><div style="font-size: 2rem; font-weight: 700; color: #10b981; margin: 0.5rem 0;">{avg_price:,} PLN</div></div>""",
                unsafe_allow_html=True)
        with col4:
            if apartment_routes:
                apt_ids = final_filtered_df[
                    'listing_id'].tolist() if 'listing_id' in final_filtered_df.columns else final_filtered_df.index.tolist()
                travel_times = [route.get('total_time', 0) for route in apartment_routes.values() if
                                route['listing_id'] in apt_ids]
                if travel_times:
                    avg_travel = int(np.mean(travel_times))
                    st.markdown(
                        f"""<div class="metric-card"><div style="font-size: 0.9rem; color: #718096; font-weight: 500; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">Avg Travel Time</div><div style="font-size: 2rem; font-weight: 700; color: #8b5cf6; margin: 0.5rem 0;">{avg_travel} min</div></div>""",
                        unsafe_allow_html=True)

        selected_rooms_clean = [int(room) for room in filters['selected_rooms']]
        st.markdown(
            f"""<div class="info-box"><strong>📅 Active Filters:</strong> Rooms: {selected_rooms_clean} | Max Price: {filters['max_price']:,} PLN | Max Travel: {filters['max_travel_time']} min | Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</div>""",
            unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📋 List View", "📊 Analytics"])
        with tab1:
            st.markdown("### 🗺️ Apartment Locations")
            if st.session_state.map_needs_update or st.session_state.map_data is None:
                map_data = analyzer.create_map_visualization(final_filtered_df, apartment_routes, new_today_listings)
                if st.session_state.temp_location and map_data is not None:
                    temp_loc = st.session_state.temp_location
                    lat, lon = temp_loc['coordinates']
                    color, icon, popup_text = ('#AEC6CF', 'check',
                                               f"✅ FEASIBLE<br>{temp_loc['address']}<br>🕐 {temp_loc['travel_time']:.1f} min to work") if \
                    temp_loc['feasible'] else (
                    'red', 'remove', f"❌ NOT FEASIBLE<br>{temp_loc['address']}<br>{temp_loc['reason']}")
                    folium.Marker([lat, lon], popup=folium.Popup(popup_text, max_width=300),
                                  icon=folium.Icon(color=color, icon=icon), tooltip="Address Check Result").add_to(
                        map_data)
                st.session_state.map_data = map_data
                st.session_state.map_needs_update = False
            if st.session_state.map_data is not None:
                st_folium(st.session_state.map_data, width=None, height=900, returned_objects=[], key="main_map")
        with tab2:
            st.markdown("### 📋 Apartment Details")
            display_df = final_filtered_df.sort_values(['is_new_today', 'price'], ascending=[False, True])
            if 'listing_id' in display_df.columns:
                for _, apartment in display_df.iterrows():
                    listing_id = apartment['listing_id']
                    route = apartment_routes.get(listing_id, {})
                    travel_time = f"{route.get('total_time', 0):.1f}" if route else "N/A"
                    is_new = apartment.get('is_new_today', False)
                    card_class = "apartment-card new-today" if is_new else "apartment-card"
                    status_emoji, days_info = ("🆕", "New Today!") if is_new else (
                    "📅", f"{apartment.get('days_since_created', 0)} days ago")
                    location = get_location_display(apartment)
                    created_date = apartment['created_at'].strftime(
                        '%Y-%m-%d') if 'created_at' in apartment and pd.notna(apartment['created_at']) else "Unknown"
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(
                            f"""<div class="{card_class}"><h3>{status_emoji} {apartment['rooms']} rooms - {apartment.get('area', 'N/A')}m²</h3><div style="color: #475569; margin-bottom: 0.5rem; font-size: 0.95rem;"><strong>📍 Location:</strong> {location}</div><div style="color: #475569; margin-bottom: 0.5rem; font-size: 0.95rem;"><strong>💰 Price:</strong> {apartment['price']:,} PLN/month</div><div style="color: #475569; margin-bottom: 0.5rem; font-size: 0.95rem;"><strong>🚌 Travel Time:</strong> {travel_time} min</div><div style="color: #475569; margin-bottom: 0.5rem; font-size: 0.95rem;"><strong>📅 Created:</strong> {created_date} ({days_info})</div><div style="color: #475569; margin-bottom: 0.5rem; font-size: 0.95rem;"><strong>🚶 Route:</strong> Walk {route.get('walking_time_from_apt', 0):.1f}min → 🚌 Transit {route.get('transit_time', 0):.0f}min → 🚶 Walk {route.get('walking_time_to_office', 0):.1f}min</div></div>""",
                            unsafe_allow_html=True)
                    with c2:
                        if 'url' in apartment and apartment['url']:
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            st.link_button("🏠 View Listing", apartment['url'], use_container_width=True)
        with tab3:
            st.markdown("### 📊 Market Analytics")

            # Use placeholders for a better loading experience, preventing screen jumps
            col1, col2 = st.columns(2)
            chart1_placeholder = col1.empty()
            chart2_placeholder = col2.empty()
            chart3_placeholder = st.empty()
            chart4_placeholder = st.empty()

            with st.spinner("🎨 Generating beautiful analytics..."):
                # Create two columns for the main distribution charts
                with chart1_placeholder.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(create_price_distribution_chart(final_filtered_df), use_container_width=True,
                                    key="price_dist")
                    st.markdown('</div>', unsafe_allow_html=True)

                with chart2_placeholder.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(create_travel_time_distribution_chart(final_filtered_df, apartment_routes),
                                    use_container_width=True, key="travel_dist")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Display the more detailed, full-width charts below
                with chart3_placeholder.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(create_price_vs_commute_scatter(final_filtered_df, apartment_routes),
                                    use_container_width=True, key="price_vs_commute")
                    st.markdown('</div>', unsafe_allow_html=True)

                with chart4_placeholder.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(create_listings_timeline_chart(final_filtered_df), use_container_width=True,
                                    key="timeline")
                    st.markdown('</div>', unsafe_allow_html=True)



    else:
        # STATE 2: NO RESULTS FOUND - Display enhanced feedback ONLY
        st.warning("🚫 **No apartments match your criteria**")

        # Check why no results were found by looking at the dataframe before the final filter
        pre_travel_filter_df = st.session_state.filtered_df.copy()
        if 'created_at_date' in pre_travel_filter_df.columns:
            start_date, end_date = filters['date_range']
            pre_travel_filter_df = pre_travel_filter_df[(pre_travel_filter_df['created_at_date'] >= start_date) & (
                        pre_travel_filter_df['created_at_date'] <= end_date)]

        if not pre_travel_filter_df.empty:
            st.info(f"💡 **Try**: Increasing max travel time above {filters['max_travel_time']} minutes.")
            if apartment_routes:
                available_times = [route.get('total_time', 0) for route in apartment_routes.values()]
                if available_times:
                    st.info(
                        f"📊 **Available options**: The shortest commute found for your other criteria is {min(available_times):.1f} minutes.")
        else:
            st.info("💡 **Try**: Adjusting your room count, budget, or date range.")

else:
    # STATE 3: INITIAL LOAD / REFRESH - Display welcome message ONLY
    # This block runs because 'filter_applied' is False by default.
    st.markdown("""
            <div class="info-box">
                <h3>🏠 Welcome to Warsaw Apartment Hunter!</h3>
                <p>Use the filters in the sidebar to find apartments that match your criteria and commute preferences.</p>
                <p><strong>Getting Started:</strong></p>
                <ul>
                    <li>Set your room and price preferences</li>
                    <li>Choose your maximum travel time to work</li>
                    <li>Select the date range for listings</li>
                    <li>Click "Find Apartments" to see results</li>
                </ul>
                <p><strong>Quick Address Check:</strong></p>
                <ul>
                    <li>Use the sidebar tool to check any Warsaw address</li>
                    <li>Get instant feasibility analysis</li>
                    <li>See travel time breakdown</li>
                    <li>View results on the map</li>
                </ul>
                <p><strong>Data Management:</strong></p>
                <ul>
                    <li>Use "Update Apartments" to run the scraper for new listings</li>
                    <li>Use "Clear Cache" to refresh the dashboard</li>
                    <li><strong>Note:</strong> Transport data is pre-optimized and doesn't need updates</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR FOOTER WITH RESULTS AND SYSTEM INFO
# ==========================================

# Sidebar footer with current results and system info
with st.sidebar:

    st.markdown("---")
    st.markdown("*Warsaw Apartment Hunter v1.0*")


# ==========================================
# DEPLOYMENT READINESS CHECK
# ==========================================

def check_deployment_readiness():
    """
    Check if the application is ready for deployment.
    This function validates that all required files exist and are properly sized.
    """
    issues = []

    # Check GTFS files
    gtfs_files = ['stops.txt', 'routes.txt', 'trips.txt', 'stop_times.txt', 'shapes.txt', 'calendar.txt']
    transport_dir = project_root / "data" / "transport"

    for file in gtfs_files:
        file_path = transport_dir / file
        if not file_path.exists():
            issues.append(f"Missing GTFS file: {file}")
        else:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > 100:  # Files should be much smaller after optimization
                issues.append(f"GTFS file too large: {file} ({size_mb:.1f} MB)")

    # Check apartment data
    apartment_file = project_root / "data" / "apartment" / "warsaw_private_owner_apartments.xlsx"
    if not apartment_file.exists():
        issues.append("Missing apartment data file")

    # Check for large cache files that shouldn't be in Git
    cache_dir = project_root / "cache"
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.pkl"):
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            if size_mb > 50:
                issues.append(
                    f"Large cache file detected: {cache_file.name} ({size_mb:.1f} MB) - should be in .gitignore")

    return issues


# Run deployment check in development mode only
if __name__ == "__main__":
    # Only run deployment check if not in production
    if not os.environ.get('STREAMLIT_SHARING_MODE'):
        deployment_issues = check_deployment_readiness()
        if deployment_issues:
            st.sidebar.warning("⚠️ Deployment Issues Detected")
            with st.sidebar.expander("View Issues"):
                for issue in deployment_issues:
                    st.write(f"• {issue}")

# ==========================================
# APPLICATION FOOTER
# ==========================================

# Add footer information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;">
    <p><strong>Warsaw Apartment Hunter</strong> - Intelligent apartment search with commute optimization</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
