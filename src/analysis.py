import os
from pathlib import Path
import pandas as pd
import numpy as np
import folium
from datetime import datetime, time, timedelta
import time as time_module
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pickle
import re
import random


class Timer:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    END = "\033[0m"

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(f"{self.CYAN}[START] {self.name} {datetime.now().strftime('%H:%M:%S')}{self.END}", flush=True)
        self.start = time_module.time()
        return self

    def __exit__(self, *args):
        elapsed = time_module.time() - self.start
        print(f"{self.GREEN}[END] {self.name} ({elapsed:.2f}s) {datetime.now().strftime('%H:%M:%S')}{self.END}",
              flush=True)


def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


class ApartmentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.work_arrival_time_early = time(8, 0)
        self.work_arrival_time_late = time(9, 0)
        self.work_departure_time_early = time(15, 0)
        self.work_departure_time_late = time(16, 0)
        # CHANGE: Use static date instead of calendar logic
        self.operating_date = datetime(2025, 6, 9).date()  # Static Tuesday
        self.shape_stop_order = {}
        self.stop_to_routes = defaultdict(list)
        self.apartment_routes = {}
        self.shape_coordinates = {}
        self.new_today_listings = set()
        with Timer("One-Time Data Initialization"):
            # CHANGE: Remove calendar loading and date selection
            # self._load_calendar()
            # self._select_operating_date()
            self._load_data_optimized()
            self._normalize_coordinate_columns()  # ADD: Coordinate normalization
            self._preprocess_stops()
            self._find_direct_route_variants()
            self._precompute_shape_stop_mappings_optimized()
            self._precompute_shape_coordinates()

    # ADD: New method to normalize coordinates
    def _normalize_coordinate_columns(self):
        """Normalize coordinate columns from scraper output to expected format"""
        with Timer("Normalizing Coordinate Columns"):
            if 'latitude' in self.apartments.columns and 'longitude' in self.apartments.columns:
                self.apartments['lat'] = pd.to_numeric(self.apartments['latitude'], errors='coerce')
                self.apartments['lon'] = pd.to_numeric(self.apartments['longitude'], errors='coerce')
                print(f"Converted coordinates for {len(self.apartments)} apartments")

    def run_interactive_analysis(self, max_price, allowed_rooms):
        with Timer("Running Interactive Analysis"):
            # Always reload apartments data to get latest from Excel
            self.apartments = self.load_apartments_data()

            # ADD: Normalize coordinates after loading
            self._normalize_coordinate_columns()

            # Calculate days since created and identify new listings
            self.apartments, self.new_today_listings = self.calculate_days_since_created(self.apartments)

            self._filter_apartments(max_price, allowed_rooms)
            self._precompute_apartment_routes_ultra_optimized()

        # Return both apartment_routes and new_today_listings
        return self.apartment_routes, self.new_today_listings

    def load_apartments_with_calamine_and_smart_cache(self):
        """Load apartments using calamine with file modification checking"""
        excel_path = self.config['apartments_path']
        cache_file = Path('cache/apartments_calamine.pkl')
        cache_file.parent.mkdir(exist_ok=True)

        # Check if Excel file has been modified
        excel_mod_time = os.path.getmtime(excel_path) if os.path.exists(excel_path) else 0
        cache_mod_time = os.path.getmtime(cache_file) if cache_file.exists() else 0

        # Use cache only if it's newer than Excel file and less than 1 hour old
        cache_age = time_module.time() - cache_mod_time
        if cache_file.exists() and cache_mod_time > excel_mod_time and cache_age < 3600:  # 1 hour = 3600 seconds
            print("Loading apartments from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # Load fresh data using calamine
        print("Loading fresh apartments data with calamine...")
        try:
            apartments = pd.read_excel(
                excel_path,
                sheet_name=self.config.get('apartments_sheet', 'Private_Owner_Apartments'),
                engine="calamine"  # Use fast calamine engine
            )
        except Exception as e:
            print(f"Calamine failed ({e}), falling back to openpyxl...")
            apartments = pd.read_excel(
                excel_path,
                sheet_name=self.config.get('apartments_sheet', 'Private_Owner_Apartments'),
                engine="openpyxl"
            )

        # Cache the processed data
        with open(cache_file, 'wb') as f:
            pickle.dump(apartments, f)

        return apartments

    def load_apartments_data(self):
        """Optimized loading with calamine and smart caching"""
        with Timer("Loading Apartments Data"):
            try:
                # First try calamine with caching
                apartments = self.load_apartments_with_calamine_and_smart_cache()

                # Data processing and validation
                apartments = apartments.dropna(subset=['listing_id'])
                apartments['price'] = pd.to_numeric(apartments['price'], errors='coerce')
                apartments['rooms'] = pd.to_numeric(apartments['rooms'], errors='coerce')
                apartments['area'] = pd.to_numeric(apartments['area'], errors='coerce')

                print(f"Loaded {len(apartments)} apartments using optimized method")
                return apartments

            except Exception as e:
                print(f"Error loading with optimized method: {e}")
                # Fallback to standard pandas
                return pd.read_excel(self.config['apartments_path'],
                                     sheet_name=self.config.get('apartments_sheet', 'Private_Owner_Apartments'))

    def create_map_visualization(self, apartments_to_display, apartment_routes, new_today_listings=None):
        """
        Create map visualization with proper new listing highlighting
        new_today_listings should be a set of listing_ids that were added today
        """
        with Timer("Map Visualization"):
            if new_today_listings is None:
                new_today_listings = set()

            # Calculate days since created for display
            apartments_to_display, _ = self.calculate_days_since_created(apartments_to_display)

            m = folium.Map(location=[self.config['office_lat'], self.config['office_lon']], zoom_start=13,
                           tiles='OpenStreetMap')
            self._add_complete_route_shapes_to_map(m)
            self._add_relevant_stops_to_map(m, apartment_routes)
            folium.Marker([self.config['office_lat'], self.config['office_lon']], popup="Work Office",
                          icon=folium.Icon(color='black', icon='briefcase')).add_to(m)

            for apt in apartments_to_display.itertuples():
                route = apartment_routes.get(apt.listing_id)
                if not route or 'path_coords' not in route or len(route['path_coords']) < 2:
                    continue

                folium.PolyLine(route['path_coords'][:2], color='gray', weight=3, opacity=0.8, dash_array='5, 5',
                                popup=f"Walk to {route['boarding_stop']}").add_to(m)

                # Check if this apartment is new today
                is_new_today = apt.listing_id in new_today_listings

                # Updated popup to reflect new vs old status
                status_text = "New Today! 🆕" if is_new_today else f"Created: {getattr(apt, 'days_since_created', 'unknown')} days ago"

                popup_html = f"""
                <div data-listing-id="{apt.listing_id}" style='font-family: Arial; font-size: 12px; width: 320px; padding: 10px;'>
                    <h4 style='margin-bottom: 10px; color: #2c3e50;'>{apt.price:,.0f} PLN - {apt.rooms} rooms</h4>
                    <div style='margin-bottom: 8px;'><strong>🚶 Walk to Stop:</strong> {route['walking_time_from_apt']:.1f} min</div>
                    <div style='margin-bottom: 8px;'><strong>🚌 Route:</strong> {route['route_short_name']} ({route['route_type']}) for {route['transit_time']:.0f} min</div>
                    <div style='margin-bottom: 8px;'><strong>🚶 Walk to Office:</strong> {route['walking_time_to_office']:.1f} min</div>
                    <div style='margin-bottom: 12px;'><strong>🕐 Total time:</strong> {route['total_time']:.1f} min</div>
                    <div style='text-align: center; margin-top: 10px;'><a href="{getattr(apt, 'url', '')}" target="_blank" style='background-color: #3498db; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; font-weight: bold;'>🏠 View Listing</a></div>
                    <p style='text-align:center; font-size:11px; color:{"#28a745" if is_new_today else "#6c757d"}; margin-top:10px; font-weight:bold;'>
                        {status_text}
                    </p>
                </div>"""

                # Use the new marker style logic
                marker_color, icon_name = self.get_marker_style(apt, is_new_today)
                # CHANGE: Use lat/lon instead of latitude/longitude
                marker = folium.Marker([apt.lat, apt.lon],
                                       popup=folium.Popup(popup_html, max_width=350),
                                       icon=folium.Icon(color=marker_color, icon=icon_name, prefix='fa'))

                marker._name = f"marker_{apt.listing_id}"
                marker.add_to(m)

            return m

    # REMOVE: Calendar loading method
    # def _load_calendar(self):
    #     with Timer("Calendar Processing"):
    #         calendar_path = self.config['calendar_path']
    #         calendar_dates_path = self.config['calendar_dates_path']
    #         if not Path(calendar_path).exists() and Path(calendar_dates_path).exists():
    #             calendar_dates = pd.read_csv(calendar_dates_path)
    #             self.calendar = pd.DataFrame({"service_id": ["1"], "start_date": [
    #                 pd.to_datetime(calendar_dates['date'].min(), format='%Y%m%d', errors='coerce')], "end_date": [
    #                 pd.to_datetime(calendar_dates['date'].max(), format='%Y%m%d', errors='coerce')], "monday": [1],
    #                                           "tuesday": [1], "wednesday": [1], "thursday": [1], "friday": [1],
    #                                           "saturday": [1], "sunday": [1]})
    #             self.calendar.to_csv(calendar_path, index=False)
    #         else:
    #             self.calendar = pd.read_csv(calendar_path, dtype={'service_id': str})
    #         self.calendar['start_date'] = pd.to_datetime(self.calendar['start_date'].astype(str))
    #         self.calendar['end_date'] = pd.to_datetime(self.calendar['end_date'].astype(str))

    # REMOVE: Date selection method
    # def _select_operating_date(self):
    #     with Timer("Date Selection"):
    #         today = datetime.today()
    #         valid_dates = [d.date() for cal in self.calendar.itertuples() if cal.friday == 1 for d in
    #                        pd.date_range(max(cal.start_date, today - timedelta(days=30)),
    #                                      min(cal.end_date, today + timedelta(days=30))) if d.weekday() < 5]
    #         self.operating_date = random.choice(list(set(valid_dates))) if valid_dates else today.date()
    #         print(f"Selected operating date: {self.operating_date.strftime('%Y-%m-%d')}", flush=True)

    def _load_data_optimized(self):
        with Timer("Data Loading (Optimized)"):
            # Always reload apartments data to get latest from Excel
            self.apartments = self.load_apartments_data()

            # Keep caching only for GTFS data (stops, routes, etc.) as it doesn't change frequently
            cache_file = Path('cache/gtfs_data.pkl')
            cache_file.parent.mkdir(exist_ok=True)

            # Use 1-hour cache for GTFS data too
            if cache_file.exists() and (time_module.time() - os.path.getmtime(cache_file) < 36000):  # 1 hour
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.stops, self.stop_times, self.trips, self.routes, self.shapes = (data[k] for k in
                                                                                     ['stops', 'stop_times', 'trips',
                                                                                      'routes', 'shapes'])
            else:
                # Load GTFS data
                dtypes = {'stop_id': str, 'route_id': str, 'trip_id': str, 'shape_id': str}

                def load_file(file_info):
                    return file_info[1], pd.read_csv(self.config[file_info[0]], dtype=dtypes)

                with ThreadPoolExecutor(max_workers=5) as executor:
                    for attr, data in executor.map(load_file,
                                                   [('stops_path', 'stops'), ('stop_times_path', 'stop_times'),
                                                    ('trips_path', 'trips'), ('routes_path', 'routes'),
                                                    ('shapes_path', 'shapes')]):
                        setattr(self, attr, data)

                # Parse times and save GTFS cache
                def parse_times_vectorized(time_series):
                    time_parts = time_series.str.split(':', expand=True).astype(int)
                    hours, minutes, seconds = time_parts[0], time_parts[1], time_parts[2].fillna(0)
                    day_offset = (hours >= 24).astype(int)
                    hours %= 24
                    time_strs = hours.astype(str).str.zfill(2) + ':' + minutes.astype(str).str.zfill(
                        2) + ':' + seconds.astype(str).str.zfill(2)
                    return pd.to_datetime(self.operating_date.strftime('%Y-%m-%d ') + time_strs,
                                          format='%Y-%m-%d %H:%M:%S',
                                          errors='coerce') + pd.to_timedelta(day_offset, unit='D')

                self.stop_times['departure_time'] = parse_times_vectorized(self.stop_times['departure_time'])
                self.stop_times['arrival_time'] = parse_times_vectorized(self.stop_times['arrival_time'])
                self.stop_times.dropna(subset=['departure_time', 'arrival_time'], inplace=True)

                # Save only GTFS data to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump({'stops': self.stops, 'stop_times': self.stop_times,
                                 'trips': self.trips, 'routes': self.routes, 'shapes': self.shapes}, f)

    def calculate_days_since_created(self, apartments_df):
        """Calculate days since each apartment was created"""
        with Timer("Calculating Days Since Created"):
            current_date = pd.Timestamp.now().normalize()

            # Handle created_at column
            if 'created_at' in apartments_df.columns:
                apartments_df['created_at'] = pd.to_datetime(apartments_df['created_at'], errors='coerce')
                apartments_df['days_since_created'] = (
                        current_date - apartments_df['created_at'].dt.normalize()).dt.days
            else:
                # If no created_at column, assume all are old
                apartments_df['days_since_created'] = 999

            # Identify new today listings (created_at is today)
            new_today_mask = apartments_df['days_since_created'] == 0
            new_today_listings = set(apartments_df[new_today_mask]['listing_id'].tolist())

            return apartments_df, new_today_listings

    def get_date_range_for_apartments(self):
        """Get the date range available in the apartments data"""
        if 'created_at' in self.apartments.columns:
            self.apartments['created_at'] = pd.to_datetime(self.apartments['created_at'], errors='coerce')
            valid_dates = self.apartments['created_at'].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                return min_date, max_date

        # Fallback to current date range if no valid dates
        today = datetime.now().date()
        return today - timedelta(days=30), today

    def _preprocess_stops(self):
        with Timer("Stop Processing (CPU)"):
            valid_stops = self.stops.dropna(subset=['stop_lat', 'stop_lon']).copy()
            distances = haversine_np(valid_stops['stop_lat'].values, valid_stops['stop_lon'].values,
                                     self.config['office_lat'], self.config['office_lon'])
            valid_stops['distance'] = distances
            self.stops = self.stops.merge(valid_stops[['stop_id', 'distance']], on='stop_id', how='left')
            self.work_stops = self.stops[(self.stops['distance'] <= self.config['bus_radius_m']) | (
                    self.stops['distance'] <= self.config['tram_radius_m'])].dropna(subset=['distance'])
            self.work_stop_ids = set(self.work_stops['stop_id'])
            print(f"Found {len(self.work_stop_ids)} work stops within range")

    def _find_direct_route_variants(self):
        with Timer("Route Variant Identification"):
            variants_list = []
            for route_type, radius in [(3, self.config['bus_radius_m']), (0, self.config['tram_radius_m'])]:
                stop_ids = set(
                    self.stops[(self.stops['distance'] <= radius) & self.stops['distance'].notna()]['stop_id'])
                if not stop_ids:
                    continue
                st_trips_routes = self.stop_times[self.stop_times['stop_id'].isin(stop_ids)].merge(self.trips,
                                                                                                   on='trip_id').merge(
                    self.routes, on='route_id')
                variants = st_trips_routes[(st_trips_routes['route_type'] == route_type) & (
                        st_trips_routes['arrival_time'].dt.time >= self.work_arrival_time_early) & (st_trips_routes[
                                                                                                        'arrival_time'].dt.time <= self.work_arrival_time_late)][
                    ['route_id', 'shape_id', 'route_short_name', 'trip_headsign', 'route_type']].drop_duplicates()
                variants_list.append(variants)
            self.route_variants = pd.concat(variants_list,
                                            ignore_index=True).drop_duplicates() if variants_list else pd.DataFrame()
            self.direct_route_ids = set(self.route_variants['route_id'])
            print(f"Found {len(self.direct_route_ids)} direct routes")

    def _filter_apartments(self, max_price, allowed_rooms):
        with Timer("Apartment Filtering"):
            # CHANGE: Use lat/lon instead of latitude/longitude
            numeric_cols = ['lat', 'lon', 'price', 'rooms']
            for col in numeric_cols:
                self.apartments[col] = pd.to_numeric(self.apartments[col], errors='coerce')
            self.apartments.dropna(subset=numeric_cols, inplace=True)
            self.filtered_apartments = self.apartments[
                (self.apartments['price'] <= max_price) & (self.apartments['rooms'].isin(allowed_rooms))].copy()
            print(f"After filtering: {len(self.filtered_apartments)} apartments remain")

    def _precompute_shape_stop_mappings_optimized(self):
        with Timer("Shape-Stop Mappings (Optimized)"):
            if self.route_variants.empty:
                return
            route_shape_pairs = list(zip(self.route_variants['route_id'], self.route_variants['shape_id']))
            trips_subset = self.trips[['route_id', 'shape_id', 'trip_id']].copy()
            trips_subset['route_shape'] = list(zip(trips_subset['route_id'], trips_subset['shape_id']))
            relevant_trips = trips_subset[trips_subset['route_shape'].isin(route_shape_pairs)]
            if relevant_trips.empty:
                return
            trip_per_route_shape = relevant_trips.groupby(['route_id', 'shape_id']).first().reset_index()
            stop_sequences = self.stop_times[self.stop_times['trip_id'].isin(set(trip_per_route_shape['trip_id']))][
                ['trip_id', 'stop_id', 'stop_sequence']].sort_values(['trip_id', 'stop_sequence']).groupby('trip_id')[
                'stop_id'].apply(list).to_dict()
            for _, row in trip_per_route_shape.iterrows():
                key = (row['route_id'], row['shape_id'])
                if row['trip_id'] in stop_sequences:
                    stops_seq = stop_sequences[row['trip_id']]
                    self.shape_stop_order[key] = stops_seq
                    for stop_id in stops_seq:
                        self.stop_to_routes[stop_id].append(key)

    def _precompute_shape_coordinates(self):
        with Timer("Shape Coordinates Processing"):
            used_shape_ids = {shape_id for _, shape_id in self.shape_stop_order.keys()}
            if not used_shape_ids:
                return
            for shape_id, group in self.shapes[self.shapes['shape_id'].isin(used_shape_ids)].groupby('shape_id'):
                self.shape_coordinates[shape_id] = group.sort_values('shape_pt_sequence')[
                    ['shape_pt_lat', 'shape_pt_lon']].values.tolist()

    def _find_nearby_stops(self, apartment_lat, apartment_lon, max_distance):
        valid_stops = self.stops.dropna(subset=['stop_lat', 'stop_lon'])
        distances = haversine_np(valid_stops['stop_lat'].values, valid_stops['stop_lon'].values, apartment_lat,
                                 apartment_lon)
        nearby_stops_df = valid_stops[distances <= max_distance].copy()
        nearby_stops_df['distance'] = distances[distances <= max_distance]
        return list(zip(nearby_stops_df['stop_id'], nearby_stops_df['distance']))

    def _precompute_apartment_routes_ultra_optimized(self):
        with Timer("Route Precomputation (Ultra-Optimized Multi-Stop)"):
            if self.filtered_apartments.empty:
                self.apartment_routes = {}
                return
            self.apartment_routes = {}
            work_arrivals = self.stop_times[(self.stop_times['stop_id'].isin(self.work_stop_ids)) & (
                    self.stop_times['arrival_time'].dt.time >= self.work_arrival_time_early) & (self.stop_times[
                                                                                                    'arrival_time'].dt.time <= self.work_arrival_time_late) & (
                                                self.stop_times['arrival_time'].notna())]
            valid_trip_ids = set(work_arrivals['trip_id'])
            if not valid_trip_ids:
                return
            route_metadata = dict(
                zip(self.routes['route_id'], zip(self.routes['route_short_name'], self.routes['route_type'])))
            stop_coords = dict(zip(self.stops['stop_id'], zip(self.stops['stop_lat'], self.stops['stop_lon'])))
            trip_lookup = dict(zip(self.trips['trip_id'], zip(self.trips['route_id'], self.trips['shape_id'])))
            valid_stop_times = self.stop_times[self.stop_times['trip_id'].isin(valid_trip_ids)][
                ['trip_id', 'stop_id', 'departure_time', 'arrival_time']].copy()
            trip_stop_times = {trip_id: dict(zip(group['stop_id'], zip(group['departure_time'], group['arrival_time'])))
                               for trip_id, group in valid_stop_times.groupby('trip_id')}
            route_work_stops = {(route_id, shape_id): [ws for ws in self.work_stop_ids if ws in stops_seq] for
                                (route_id, shape_id), stops_seq in self.shape_stop_order.items()}
            route_valid_trips = defaultdict(list)
            for trip_id in valid_trip_ids:
                if trip_id in trip_lookup:
                    route_valid_trips[trip_lookup[trip_id]].append(trip_id)

            for apt in self.filtered_apartments.itertuples():
                # CHANGE: Use lat/lon instead of latitude/longitude
                nearby_stops = self._find_nearby_stops(apt.lat, apt.lon,
                                                       self.config['apartment_max_distance'])
                if not nearby_stops:
                    continue
                best_route = None
                min_total_time = float('inf')
                for stop_id, walking_dist_from_apt in nearby_stops:
                    for route_id, shape_id in self.stop_to_routes.get(stop_id, []):
                        stops_seq = self.shape_stop_order.get((route_id, shape_id), [])
                        if not stops_seq:
                            continue
                        try:
                            stop_idx = stops_seq.index(stop_id)
                        except ValueError:
                            continue
                        valid_work_stops = [ws for ws in route_work_stops.get((route_id, shape_id), []) if
                                            stops_seq.index(ws) > stop_idx]
                        if not valid_work_stops:
                            continue

                        for work_stop in valid_work_stops:
                            for trip_id in route_valid_trips.get((route_id, shape_id), []):
                                trip_data = trip_stop_times.get(trip_id, {})
                                if stop_id in trip_data and work_stop in trip_data:
                                    departure, arrival = trip_data[stop_id][0], trip_data[work_stop][1]
                                    transit_time_mins = (arrival - departure).total_seconds() / 60
                                    walking_time_from_apt_mins = walking_dist_from_apt / 83.33

                                    work_stop_coords = stop_coords.get(work_stop)
                                    if not work_stop_coords:
                                        continue
                                    walking_dist_to_office = haversine_np(work_stop_coords[0], work_stop_coords[1],
                                                                          self.config['office_lat'],
                                                                          self.config['office_lon'])
                                    walking_time_to_office_mins = walking_dist_to_office / 83.33

                                    total_time = walking_time_from_apt_mins + transit_time_mins + walking_time_to_office_mins

                                    if 0 < transit_time_mins < 120 and total_time < min_total_time:
                                        min_total_time = total_time
                                        route_short_name, route_type_code = route_metadata.get(route_id, ('Unknown', 3))
                                        best_route = {
                                            'listing_id': apt.listing_id,
                                            'departure_time': departure.strftime('%H:%M'),
                                            'arrival_time': arrival.strftime('%H:%M'),
                                            'transit_time': transit_time_mins,
                                            'walking_time_from_apt': walking_time_from_apt_mins,
                                            'walking_time_to_office': walking_time_to_office_mins,
                                            'total_time': total_time,
                                            'route_short_name': route_short_name,
                                            'route_type': 'Bus' if route_type_code == 3 else 'Tram',
                                            'boarding_stop': stop_id,
                                            'destination_stop': work_stop,
                                            'route_id': route_id,
                                            'shape_id': shape_id,
                                            'walking_distance': walking_dist_from_apt
                                        }
                if best_route:
                    # CHANGE: Use lat/lon instead of latitude/longitude
                    best_route['path_coords'] = [[apt.lat, apt.lon]] + (
                        [list(stop_coords[best_route['boarding_stop']])] if best_route[
                                                                                'boarding_stop'] in stop_coords else [])
                    self.apartment_routes[apt.listing_id] = best_route

    def _add_complete_route_shapes_to_map(self, m):
        with Timer("Adding Complete Route Shapes"):
            work_serving_routes = set()
            work_trips = set(self.stop_times[self.stop_times['stop_id'].isin(self.work_stop_ids)]['trip_id'])
            for _, trip in self.trips[self.trips['trip_id'].isin(work_trips)].iterrows():
                if pd.notna(trip['shape_id']):
                    work_serving_routes.add((trip['route_id'], trip['shape_id']))
            for route_id, shape_id in work_serving_routes:
                route_info = self.routes[self.routes['route_id'] == route_id].iloc[0]
                color = '#FF6961' if route_info['route_type'] == 0 else 'AEC6CF'
                shape_coords = self.shape_coordinates.get(shape_id)
                if shape_coords and len(shape_coords) > 1:
                    folium.PolyLine(locations=shape_coords, color=color, weight=3, opacity=0.6,
                                    popup=f"Route {route_info['route_short_name']}").add_to(m)

    def _add_relevant_stops_to_map(self, m, apartment_routes):
        with Timer("Adding Relevant Stops"):
            boarding_stops = {route['boarding_stop'] for route in apartment_routes.values()}
            stops_to_plot = self.stops[self.stops['stop_id'].isin(self.work_stop_ids.union(boarding_stops))]
            for stop in stops_to_plot.itertuples():
                color = 'darkred' if stop.stop_id in self.work_stop_ids else 'gray'
                folium.CircleMarker([stop.stop_lat, stop.stop_lon], radius=4, color=color, fill=True, fill_color=color,
                                    popup=stop.stop_name).add_to(m)

    def get_marker_style(self, apartment, is_new_today):
        """
        Determine marker color and icon based on created_at date and travel time.
        is_new_today: A boolean indicating if the apartment is new today.
        """
        if is_new_today:
            # New today - always green regardless of travel time
            return 'green', 'home'
        else:
            # Older listings - color by travel time
            listing_id = getattr(apartment, 'listing_id', apartment.name if hasattr(apartment, 'name') else None)
            route = self.apartment_routes.get(listing_id, {})
            total_time = route.get('total_time', float('inf'))

            if total_time <= 30:
                return 'green', 'home'
            elif total_time <= 45:
                return 'orange', 'home'
            else:
                return 'red', 'home'
