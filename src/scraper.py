import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from datetime import datetime
import os
from urllib.parse import urljoin, urlparse, parse_qs
import json
import sys
import os

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class OtodomScraper:
    def __init__(self):
        self.base_url = "https://www.otodom.pl"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Fix path to point to data/apartment folder
        from pathlib import Path
        script_dir = Path(__file__).parent.resolve()  # This is src folder
        project_root = script_dir.parent  # Go up one level to project root
        self.excel_file = str(project_root / "data" / "apartment" / "warsaw_private_owner_apartments.xlsx")
        self.sheet_name = "Private_Owner_Apartments"

        print(f"Excel file path: {self.excel_file}")

    def get_page_content(self, url):
        """Get page content with error handling"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def should_skip_listing(self, soup, json_data):
        """Check if listing should be skipped - only skip if no street found anywhere"""
        if not soup:
            return True

        # First, try to extract street from all possible sources
        street_found = False

        # Try JSON data first
        if json_data:
            try:
                ad_data = json_data['props']['pageProps']['ad']
                location = ad_data.get('location', {})
                address = location.get('address', {})
                street_obj = address.get('street')

                if street_obj and street_obj.get('name'):
                    street_found = True
                    print(f"  -> Street found in JSON: {street_obj['name']}")
            except:
                pass

        # If no street in JSON, try comprehensive text search
        if not street_found:
            # Get all text from the page
            page_text = soup.get_text()

            # Try to find street patterns in the entire page
            street_from_page = self.extract_complete_address_from_text(page_text)
            if street_from_page:
                street_found = True
                print(f"  -> Street found in page: {street_from_page}")

        # Enhanced fallback - try additional selectors and methods
        if not street_found:
            street_found = self.enhanced_address_fallback(soup)

        if not street_found:
            print("  -> Skipping: No street name found anywhere")
            return True

        return False

    def enhanced_address_fallback(self, soup):
        """Enhanced fallback method for address extraction"""
        # Try additional selectors for address information
        address_selectors = [
            '[data-testid="ad-location"]',
            '[class*="location"]',
            '[class*="address"]',
            '[class*="street"]',
            '.offer-item__location',
            '.location-text',
            '.address-text',
            'span[title*="ul."]',
            'span[title*="al."]',
            'div[title*="ul."]',
            'div[title*="al."]',
            '[aria-label*="lokalizacja"]',
            '[aria-label*="adres"]'
        ]

        for selector in address_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text:
                    address = self.extract_complete_address_from_text(text)
                    if address:
                        print(f"  -> Enhanced fallback found address: {address}")
                        return True

        # Try to find address in meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            content = meta.get('content', '')
            if content:
                address = self.extract_complete_address_from_text(content)
                if address:
                    print(f"  -> Found address in meta tag: {address}")
                    return True

        # Try to find address in data attributes
        elements_with_data = soup.find_all(attrs=lambda x: x and any(key.startswith('data-') for key in x.keys()))
        for element in elements_with_data:
            for attr, value in element.attrs.items():
                if attr.startswith('data-') and isinstance(value, str):
                    address = self.extract_complete_address_from_text(value)
                    if address:
                        print(f"  -> Found address in data attribute {attr}: {address}")
                        return True

        return False

    def extract_listing_urls(self, page_url):
        """Extract all listing URLs from a search results page"""
        soup = self.get_page_content(page_url)
        if not soup:
            return []

        listing_urls = []
        articles = soup.find_all('article')

        for article in articles:
            link = article.find('a', href=True)
            if link:
                relative_url = link['href']
                if '/pl/oferta/' in relative_url:
                    full_url = urljoin(self.base_url, relative_url)
                    listing_urls.append(full_url)

        return listing_urls

    def extract_listing_id(self, url):
        """Extract listing ID from URL"""
        match = re.search(r'-ID([a-zA-Z0-9]+)$', url)
        return f"otodom_{match.group(1)}" if match else None

    def extract_listing_data(self, url):
        """Extract all required data from a single listing page"""
        soup = self.get_page_content(url)
        if not soup:
            return None

        # Extract listing ID
        listing_id = self.extract_listing_id(url)
        if not listing_id:
            return None

        # Extract the main JSON data
        json_data = self.extract_json_data(soup)

        # Check if this listing should be skipped (only if no street found anywhere)
        if self.should_skip_listing(soup, json_data):
            return "SKIP"

        # Extract title
        title_elem = soup.find('h1')
        title = title_elem.text.strip() if title_elem else ""

        # Extract all data using the JSON data and soup
        price = self.extract_price_from_json(json_data)
        rooms = self.extract_rooms_from_json(json_data)
        area = self.extract_area_from_json(json_data)
        address_data = self.extract_address_comprehensive(json_data, soup)
        latitude, longitude = self.extract_coordinates_from_json(json_data)

        return {
            'listing_id': listing_id,
            'title': title,
            'price': price,
            'rooms': rooms,
            'area': area,
            'raw_address': address_data.get('raw_address', ''),
            'extracted_street': address_data.get('street', ''),
            'extracted_district': '',  # Leave blank as requested
            'url': url,
            'source': 'Otodom_PrivateOwner',
            'owner_type': 'private',
            'latitude': latitude,
            'longitude': longitude,
            'geocoding_status': 'real_street' if latitude and longitude else '',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def extract_json_data(self, soup):
        """Extract the main JSON data from __NEXT_DATA__ script"""
        script = soup.find('script', {'id': '__NEXT_DATA__'})
        if script and script.string:
            try:
                return json.loads(script.string)
            except:
                pass
        return None

    def extract_price_from_json(self, json_data):
        """Extract price from JSON data"""
        if not json_data:
            return None

        try:
            ad_data = json_data['props']['pageProps']['ad']

            # Try to get price from characteristics
            characteristics = ad_data.get('characteristics', [])
            for char in characteristics:
                if char.get('key') == 'price':
                    return int(char.get('value', 0))

            # Fallback to target data
            target = ad_data.get('target', {})
            price = target.get('Price')
            if price:
                return int(price)

        except:
            pass

        return None

    def extract_rooms_from_json(self, json_data):
        """Extract number of rooms from JSON data"""
        if not json_data:
            return None

        try:
            ad_data = json_data['props']['pageProps']['ad']

            # Try to get rooms from characteristics
            characteristics = ad_data.get('characteristics', [])
            for char in characteristics:
                if char.get('key') == 'rooms_num':
                    return int(char.get('value', 0))

            # Fallback to target data
            target = ad_data.get('target', {})
            rooms_list = target.get('Rooms_num', [])
            if rooms_list and isinstance(rooms_list, list):
                return int(rooms_list[0])
            elif rooms_list:
                return int(rooms_list)

        except:
            pass

        return None

    def extract_area_from_json(self, json_data):
        """Extract area from JSON data"""
        if not json_data:
            return None

        try:
            ad_data = json_data['props']['pageProps']['ad']

            # Try to get area from characteristics
            characteristics = ad_data.get('characteristics', [])
            for char in characteristics:
                if char.get('key') == 'm':
                    return float(char.get('value', 0))

            # Fallback to target data
            target = ad_data.get('target', {})
            area = target.get('Area')
            if area:
                return float(area)

        except:
            pass

        return None

    def extract_address_comprehensive(self, json_data, soup):
        """Comprehensive address extraction from all sources with focus on complete addresses"""
        address_data = {'raw_address': '', 'street': ''}

        # Method 1: Try JSON data first
        if json_data:
            try:
                ad_data = json_data['props']['pageProps']['ad']
                location = ad_data.get('location', {})
                address = location.get('address', {})
                street_obj = address.get('street')

                if street_obj and street_obj.get('name'):
                    street_name = street_obj['name']
                    street_clean = self.clean_street_name(street_name)

                    # Try to get house number from JSON
                    house_number = street_obj.get('number', '')
                    if house_number:
                        full_street = f"{street_clean} {house_number}"
                    else:
                        full_street = street_clean

                    address_data['street'] = full_street

                    # Build raw address from location data
                    city = address.get('city', {}).get('name', '')
                    district_obj = address.get('district')
                    district = district_obj.get('name', '') if district_obj else ''

                    raw_parts = [f"ul. {full_street}"]
                    if district:
                        raw_parts.append(district)
                    if city:
                        raw_parts.append(city)

                    address_data['raw_address'] = ', '.join(raw_parts)
                    print(f"  -> Method 1 (JSON) - Found complete street: {full_street}")
                    return address_data
            except:
                pass

        # Method 2: Enhanced page text extraction with multiple approaches
        page_text = soup.get_text()
        complete_address = self.extract_complete_address_from_text(page_text)
        if complete_address:
            address_data['street'] = complete_address
            address_data['raw_address'] = f"ul. {complete_address}, Warszawa"
            print(f"  -> Method 2 (Page text) - Found complete address: {complete_address}")
            return address_data

        # Method 2.5: Enhanced selector-based extraction
        enhanced_address = self.extract_address_from_enhanced_selectors(soup)
        if enhanced_address:
            address_data['street'] = enhanced_address
            address_data['raw_address'] = f"ul. {enhanced_address}, Warszawa"
            print(f"  -> Method 2.5 (Enhanced selectors) - Found address: {enhanced_address}")
            return address_data

        # Method 3: Extract from description with enhanced patterns
        description_selectors = [
            '[data-cy="adPageDescription"]',
            '.css-1o565rw',
            '.description-text',
            '.offer-description',
            '[class*="description"]',
            '[class*="desc"]',
            '[data-testid*="description"]',
            '.ad-description',
            '.listing-description'
        ]

        description_text = ""
        for selector in description_selectors:
            desc_elem = soup.select_one(selector)
            if desc_elem:
                description_text = desc_elem.get_text()
                break

        if description_text:
            complete_address = self.extract_complete_address_from_text(description_text)
            if complete_address:
                address_data['street'] = complete_address
                address_data['raw_address'] = description_text[:200]
                print(f"  -> Method 3 (Description) - Found complete address: {complete_address}")
                return address_data

        # Method 4: Extract from title
        title_elem = soup.find('h1')
        if title_elem:
            title_text = title_elem.get_text()
            complete_address = self.extract_complete_address_from_text(title_text)
            if complete_address:
                address_data['street'] = complete_address
                address_data['raw_address'] = title_text
                print(f"  -> Method 4 (Title) - Found complete address: {complete_address}")
                return address_data

        # Method 5: Try breadcrumbs
        if json_data:
            try:
                ad_data = json_data['props']['pageProps']['ad']
                breadcrumbs = ad_data.get('breadcrumbs', [])
                for breadcrumb in breadcrumbs:
                    label = breadcrumb.get('label', '')
                    complete_address = self.extract_complete_address_from_text(label)
                    if complete_address:
                        address_data['street'] = complete_address
                        address_data['raw_address'] = label
                        print(f"  -> Method 5 (Breadcrumbs) - Found complete address: {complete_address}")
                        return address_data
            except:
                pass

        # Method 6: Enhanced meta and attribute search
        meta_address = self.extract_address_from_meta_and_attributes(soup)
        if meta_address:
            address_data['street'] = meta_address
            address_data['raw_address'] = f"ul. {meta_address}, Warszawa"
            print(f"  -> Method 6 (Meta/Attributes) - Found address: {meta_address}")
            return address_data

        # Method 7: Build raw address from available location data (no street found)
        if json_data:
            try:
                ad_data = json_data['props']['pageProps']['ad']
                location = ad_data.get('location', {})
                address = location.get('address', {})

                city = address.get('city', {}).get('name', '')
                district_obj = address.get('district')
                district = district_obj.get('name', '') if district_obj else ''

                raw_parts = []
                if district:
                    raw_parts.append(district)
                if city:
                    raw_parts.append(city)

                if raw_parts:
                    address_data['raw_address'] = ', '.join(raw_parts)
            except:
                pass

        print(f"  -> No complete address found, raw address: {address_data['raw_address']}")
        return address_data

    def extract_address_from_enhanced_selectors(self, soup):
        """Extract address using enhanced CSS selectors"""
        enhanced_selectors = [
            # Location-specific selectors
            '[data-testid="ad-location"]',
            '[data-cy="ad-location"]',
            '[class*="location"]',
            '[class*="address"]',
            '[class*="street"]',

            # Otodom-specific selectors
            '.offer-item__location',
            '.location-text',
            '.address-text',
            '.ad-location',
            '.listing-location',

            # Title and span selectors with address hints
            'span[title*="ul."]',
            'span[title*="al."]',
            'span[title*="pl."]',
            'div[title*="ul."]',
            'div[title*="al."]',
            'div[title*="pl."]',

            # Aria labels in Polish
            '[aria-label*="lokalizacja"]',
            '[aria-label*="adres"]',
            '[aria-label*="ulica"]',

            # Generic location indicators
            '.location',
            '.address',
            '.street-address',

            # Additional data attributes
            '[data-location]',
            '[data-address]',
            '[data-street]'
        ]

        for selector in enhanced_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    # Try text content
                    text = element.get_text(strip=True)
                    if text:
                        address = self.extract_complete_address_from_text(text)
                        if address:
                            return address

                    # Try title attribute
                    title = element.get('title', '')
                    if title:
                        address = self.extract_complete_address_from_text(title)
                        if address:
                            return address

                    # Try data attributes
                    for attr, value in element.attrs.items():
                        if isinstance(value, str) and (
                                'location' in attr.lower() or 'address' in attr.lower() or 'street' in attr.lower()):
                            address = self.extract_complete_address_from_text(value)
                            if address:
                                return address
            except:
                continue

        return None

    def extract_address_from_meta_and_attributes(self, soup):
        """Extract address from meta tags and data attributes"""
        # Check meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            # Check content attribute
            content = meta.get('content', '')
            if content:
                address = self.extract_complete_address_from_text(content)
                if address:
                    return address

            # Check other attributes
            for attr, value in meta.attrs.items():
                if isinstance(value, str) and len(value) > 10:
                    address = self.extract_complete_address_from_text(value)
                    if address:
                        return address

        # Check all elements with data attributes
        elements_with_data = soup.find_all(attrs=lambda x: x and any(key.startswith('data-') for key in x.keys()))
        for element in elements_with_data:
            for attr, value in element.attrs.items():
                if attr.startswith('data-') and isinstance(value, str) and len(value) > 5:
                    address = self.extract_complete_address_from_text(value)
                    if address:
                        return address

        # Check script tags for embedded JSON with location data
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            if script.string:
                try:
                    data = json.loads(script.string)
                    address = self.extract_address_from_json_ld(data)
                    if address:
                        return address
                except:
                    continue

        return None

    def extract_address_from_json_ld(self, data):
        """Extract address from JSON-LD structured data"""
        if isinstance(data, dict):
            # Look for address fields
            if 'address' in data:
                address_obj = data['address']
                if isinstance(address_obj, dict):
                    street = address_obj.get('streetAddress', '')
                    if street:
                        return self.clean_street_name(street)
                elif isinstance(address_obj, str):
                    return self.extract_complete_address_from_text(address_obj)

            # Look for location fields
            if 'location' in data:
                location = data['location']
                if isinstance(location, dict) and 'address' in location:
                    return self.extract_address_from_json_ld(location)

            # Recursively search in nested objects
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    result = self.extract_address_from_json_ld(value)
                    if result:
                        return result

        elif isinstance(data, list):
            for item in data:
                result = self.extract_address_from_json_ld(item)
                if result:
                    return result

        return None

    def extract_complete_address_from_text(self, text):
        """Extract complete address including house numbers from text with enhanced Polish patterns"""
        if not text:
            return ''

        # Enhanced patterns for complete addresses with house numbers
        complete_address_patterns = [
            # Standard patterns with house numbers - more flexible
            r'\bul\.?\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']+?)\s+(\d+[a-zA-Z]?(?:/\d+)?(?:\s*[a-zA-Z])?)\b',
            r'\bal\.?\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']+?)\s+(\d+[a-zA-Z]?(?:/\d+)?(?:\s*[a-zA-Z])?)\b',
            r'\baleja\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']+?)\s+(\d+[a-zA-Z]?(?:/\d+)?(?:\s*[a-zA-Z])?)\b',
            r'\bulica\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']+?)\s+(\d+[a-zA-Z]?(?:/\d+)?(?:\s*[a-zA-Z])?)\b',
            r'\bpl\.?\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']+?)\s+(\d+[a-zA-Z]?(?:/\d+)?(?:\s*[a-zA-Z])?)\b',
            r'\bplac\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']+?)\s+(\d+[a-zA-Z]?(?:/\d+)?(?:\s*[a-zA-Z])?)\b',
            r'\bos\.?\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']+?)\s+(\d+[a-zA-Z]?(?:/\d+)?(?:\s*[a-zA-Z])?)\b',
            r'\bosiedle\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']+?)\s+(\d+[a-zA-Z]?(?:/\d+)?(?:\s*[a-zA-Z])?)\b',

            # Patterns without explicit prefixes but with house numbers - enhanced
            r'\b([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)\s+(\d+[a-zA-Z]?(?:/\d+)?(?:\s*[a-zA-Z])?)\s+(?:w\s+)?Warszaw[aie]\b',
            r'\b([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)\s+(\d+[a-zA-Z]?(?:/\d+)?(?:\s*[a-zA-Z])?)\s*[,\.]',

            # Patterns for street names without house numbers but with context - enhanced
            r'\bul\.?\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)(?:\s+w\s|\s+na\s|\s+przy\s|\s*[,\.\n]|$)',
            r'\bal\.?\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)(?:\s+w\s|\s+na\s|\s+przy\s|\s*[,\.\n]|$)',
            r'\baleja\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)(?:\s+w\s|\s+na\s|\s+przy\s|\s*[,\.\n]|$)',
            r'\bulica\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)(?:\s+w\s|\s+na\s|\s+przy\s|\s*[,\.\n]|$)',
            r'\bpl\.?\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)(?:\s+w\s|\s+na\s|\s+przy\s|\s*[,\.\n]|$)',
            r'\bplac\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)(?:\s+w\s|\s+na\s|\s+przy\s|\s*[,\.\n]|$)',
            r'\bos\.?\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)(?:\s+w\s|\s+na\s|\s+przy\s|\s*[,\.\n]|$)',
            r'\bosiedle\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)(?:\s+w\s|\s+na\s|\s+przy\s|\s*[,\.\n]|$)',

            # Additional patterns for common Polish street formats
            r'\b([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)\s+(\d+[a-zA-Z]?(?:/\d+)?)\s*(?:m\.|mieszkanie|lok\.)?',
            r'(?:przy|na|w)\s+ul\.?\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)(?:\s+(\d+[a-zA-Z]?(?:/\d+)?))?',
            r'(?:przy|na|w)\s+al\.?\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s\-\.\']{2,}?)(?:\s+(\d+[a-zA-Z]?(?:/\d+)?))?',
        ]

        # Enhanced exclude words
        exclude_words = [
            'w', 'na', 'przy', 'obok', 'blisko', 'metrów', 'od', 'do', 'za', 'przed',
            'mieszkanie', 'apartament', 'pokój', 'wynajmę', 'oferuję', 'wynajem',
            'lokalu', 'lokale', 'nieruchomość', 'budynku', 'budynek', 'bloku', 'blok',
            'osiedlu', 'osiedle', 'centrum', 'dzielnica', 'warszawa', 'warszawie',
            'metro', 'metra', 'autobus', 'autobusu', 'tramwaj', 'tramwaju',
            'sklep', 'sklepu', 'sklepy', 'szkoła', 'szkoły', 'przedszkole',
            'parking', 'parkingu', 'garaż', 'garażu', 'balkon', 'balkonu',
            'taras', 'tarasu', 'ogród', 'ogrodu', 'piwnica', 'piwnicy',
            'kuchnia', 'kuchni', 'łazienka', 'łazienki', 'salon', 'salonu',
            'sypialnia', 'sypialni', 'pokoje', 'pokoi', 'meble', 'mebli',
            'wyposażone', 'wyposażenie', 'umeblowane', 'umeblowanie'
        ]

        # Enhanced districts and neighborhoods to exclude
        exclude_districts = [
            'mokotów', 'śródmieście', 'wola', 'ochota', 'żoliborz',
            'praga-północ', 'praga-południe', 'targówek', 'rembertów',
            'wawer', 'wilanów', 'ursynów', 'włochy', 'ursus',
            'bemowo', 'bielany', 'białołęka', 'wesoła',
            'mirów', 'muranów', 'stare miasto', 'nowe miasto', 'powiśle',
            'solec', 'ujazdów', 'frascati', 'saska kępa', 'kamionek',
            'grochów', 'gocław', 'stara ochota', 'rakowiec', 'czyste',
            'koło', 'odolany', 'ulrychów', 'stary mokotów', 'służew',
            'sadyba', 'czerniaków', 'sielce', 'kabaty', 'natolin',
            'stokłosy', 'imielin', 'grabów', 'pyry', 'jeziorki',
            'praga', 'północ', 'południe', 'centrum', 'centralny'
        ]

        for pattern in complete_address_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    # Pattern with house number
                    if len(match) == 2 and match[1]:  # Both street and house number
                        street_name, house_number = match
                        complete_address = f"{street_name.strip()} {house_number.strip()}"
                    else:
                        complete_address = match[0].strip()
                else:
                    # Pattern without house number
                    complete_address = match.strip()

                # Clean the address
                complete_address = complete_address.strip(' \t\n\r\f\v.,;:-')

                # Validate the address
                if self.validate_street_address(complete_address, exclude_words, exclude_districts):
                    return complete_address

        return ''

    def validate_street_address(self, address, exclude_words, exclude_districts):
        """Validate if the extracted address is a real street address with enhanced validation"""
        if not address or len(address) < 3 or len(address) > 100:
            return False

        address_lower = address.lower()

        # Check if it's a district name
        if address_lower in exclude_districts:
            return False

        # Check if it contains excluded words that indicate it's not a street
        if any(exclude in address_lower for exclude in
               ['mieszkanie', 'apartament', 'wynajmę', 'oferuję', 'wynajem', 'pokój', 'lokale', 'nieruchomość']):
            return False

        # Check if it starts with an excluded word
        words = address.split()
        if words and words[0].lower() in exclude_words:
            return False

        # Must contain at least one uppercase letter (proper noun)
        if not re.search(r'[A-ZĄĆĘŁŃÓŚŹŻ]', address):
            return False

        # Should not be just numbers
        if address.replace(' ', '').replace('/', '').isdigit():
            return False

        # Should not be too short after cleaning
        clean_address = re.sub(r'\d+[a-zA-Z]?(?:/\d+)?', '', address).strip()
        if len(clean_address) < 3:
            return False

        # Should contain typical Polish street name patterns
        if re.search(r'[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}', address):
            return True

        return False

    def clean_street_name(self, street_text):
        """Clean street name by removing prefixes with enhanced cleaning"""
        if not street_text:
            return ''

        # Remove common prefixes
        street_clean = re.sub(r'^(ul\.|al\.|pl\.|os\.|aleja|ulica|plac|osiedle)\s+', '', street_text.strip(),
                              flags=re.IGNORECASE)

        # Remove any trailing commas, periods, or extra whitespace
        street_clean = re.sub(r'[,\.\s]+$', '', street_clean)

        # Remove leading/trailing whitespace
        street_clean = street_clean.strip()

        return street_clean

    def extract_coordinates_from_json(self, json_data):
        """Extract coordinates from JSON data"""
        if not json_data:
            return None, None

        try:
            ad_data = json_data['props']['pageProps']['ad']
            location = ad_data.get('location', {})
            coordinates = location.get('coordinates', {})

            lat = coordinates.get('latitude')
            lon = coordinates.get('longitude')

            if lat and lon:
                return str(lat), str(lon)

        except:
            pass

        return None, None

    def load_existing_data(self):
        """Load existing data from Excel file"""
        if os.path.exists(self.excel_file):
            # Check if it's an LFS pointer file
            with open(self.excel_file, 'rb') as f:
                first_line = f.readline().decode('utf-8', errors='ignore')
                if first_line.startswith('version https://git-lfs.github.com'):
                    print(f"ERROR: Excel file is an LFS pointer, not actual content!")
                    print(f"First line: {first_line}")
                    return pd.DataFrame(columns=self.get_default_columns())

            try:
                print(f"Loading existing data from: {self.excel_file}")
                df = pd.read_excel(self.excel_file, sheet_name=self.sheet_name)
                print(f"Successfully loaded {len(df)} existing listings")
                if len(df) > 0:
                    print(f"Sample existing IDs: {df['listing_id'].head().tolist()}")
                return df
            except Exception as e:
                print(f"Error loading existing data: {e}")
                print(f"File exists: {os.path.exists(self.excel_file)}")

                # Try to load without specifying sheet name to see available sheets
                try:
                    xl_file = pd.ExcelFile(self.excel_file)
                    print(f"Available sheets: {xl_file.sheet_names}")
                except Exception as sheet_error:
                    print(f"Cannot read Excel file at all: {sheet_error}")

                # Return empty DataFrame on error
                pass
        else:
            print(f"Excel file does not exist: {self.excel_file}")

        # Return empty DataFrame with correct columns
        columns = ['listing_id', 'title', 'price', 'rooms', 'area', 'raw_address',
                   'extracted_street', 'extracted_district', 'url', 'source',
                   'owner_type', 'latitude', 'longitude', 'geocoding_status', 'created_at', 'last_seen']

        print("Returning empty DataFrame with correct columns")
        return pd.DataFrame(columns=columns)

    def get_default_columns(self):
        """Get default column structure"""
        return ['listing_id', 'title', 'price', 'rooms', 'area', 'raw_address',
                'extracted_street', 'extracted_district', 'url', 'source',
                'owner_type', 'latitude', 'longitude', 'geocoding_status', 'created_at', 'last_seen']

    def save_data(self):
        """Save data to Excel with proper handling of verified active listings"""
        from datetime import datetime, timezone
        import pandas as pd

        current_time = datetime.now(timezone.utc).replace(tzinfo=None)

        # Use updated existing data (after removals) if available
        if hasattr(self, 'existing_df_updated'):
            existing_df = self.existing_df_updated
            print(f"Using updated existing data with {len(existing_df)} listings (after removals)")
        else:
            existing_df = self.load_existing_data()
            print(f"Using original existing data with {len(existing_df)} listings")

        # Create DataFrame from new listings
        if hasattr(self, 'all_listings') and self.all_listings:
            new_df = pd.DataFrame(self.all_listings)
            new_df['last_seen'] = current_time
            print(f"Adding {len(new_df)} new listings")
        else:
            new_df = pd.DataFrame()
            print("No new listings to add")

        if not existing_df.empty:
            # Update last_seen for verified active listings
            if hasattr(self, 'verified_active_ids') and self.verified_active_ids:
                mask = existing_df['listing_id'].isin(self.verified_active_ids)
                existing_df.loc[mask, 'last_seen'] = current_time
                print(f"Updated last_seen for {mask.sum()} verified active listings")

            # Combine existing and new data
            if not new_df.empty:
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = existing_df
        else:
            # No existing data - all listings are new
            final_df = new_df if not new_df.empty else pd.DataFrame(columns=self.get_default_columns())

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.excel_file), exist_ok=True)

        # Save to Excel
        if not final_df.empty:
            with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                final_df.to_excel(writer, sheet_name=self.sheet_name, index=False)

            print(f"Data saved to {self.excel_file}")
            print(f"Total listings in final dataset: {len(final_df)}")

            # Show breakdown
            if hasattr(self, 'verified_active_ids'):
                print(f"  - Existing active: {len(self.verified_active_ids)}")
            if hasattr(self, 'all_listings'):
                print(f"  - New listings: {len(self.all_listings)}")
        else:
            print("No data to save")

    def get_all_pages(self, base_search_url):
        """Get all page URLs for the search results with improved pagination detection"""
        page_urls = [base_search_url]

        print(f"Analyzing pagination from: {base_search_url}")

        # Get first page to analyze pagination
        soup = self.get_page_content(base_search_url)
        if not soup:
            return page_urls

        # Method 1: Look for pagination in JSON data
        script = soup.find('script', {'id': '__NEXT_DATA__'})
        if script and script.string:
            try:
                json_data = json.loads(script.string)
                search_data = json_data.get('props', {}).get('pageProps', {}).get('data', {})
                pagination = search_data.get('searchAds', {}).get('pagination', {})

                total_pages = pagination.get('totalPages', 1)
                current_page = pagination.get('page', 1)

                print(f"Found pagination in JSON: Current page {current_page}, Total pages: {total_pages}")

                if total_pages > 1:
                    # Generate URLs for all pages
                    base_parts = urlparse(base_search_url)
                    base_params = parse_qs(base_parts.query)

                    for page_num in range(2, total_pages + 1):
                        # Update page parameter
                        new_params = base_params.copy()
                        new_params['page'] = [str(page_num)]

                        # Rebuild URL
                        new_query = '&'.join([f"{k}={v[0]}" for k, v in new_params.items()])
                        new_url = f"{base_parts.scheme}://{base_parts.netloc}{base_parts.path}?{new_query}"

                        page_urls.append(new_url)
                        print(f"Added page {page_num}: {new_url}")

                return page_urls

            except Exception as e:
                print(f"Error parsing JSON pagination: {e}")

        # Method 2: Look for pagination in HTML (fallback)
        pagination_selectors = [
            'nav[data-cy="frontend.search.base-pagination"]',
            '.pagination',
            '[class*="pagination"]',
            'nav[aria-label*="pagination"]',
            'nav[aria-label*="Pagination"]'
        ]

        for selector in pagination_selectors:
            pagination = soup.select_one(selector)
            if pagination:
                print(f"Found pagination with selector: {selector}")

                # Look for page links
                page_links = pagination.find_all('a', href=True)
                page_numbers = set()

                for link in page_links:
                    href = link['href']
                    # Extract page number from URL
                    if 'page=' in href:
                        try:
                            parsed_url = urlparse(href)
                            params = parse_qs(parsed_url.query)
                            if 'page' in params:
                                page_num = int(params['page'][0])
                                page_numbers.add(page_num)
                        except:
                            continue

                if page_numbers:
                    max_page = max(page_numbers)
                    print(f"Found page numbers in HTML: {sorted(page_numbers)}, max page: {max_page}")

                    # Generate URLs for all pages
                    base_parts = urlparse(base_search_url)
                    base_params = parse_qs(base_parts.query)

                    for page_num in range(2, max_page + 1):
                        # Update page parameter
                        new_params = base_params.copy()
                        new_params['page'] = [str(page_num)]

                        # Rebuild URL
                        new_query = '&'.join([f"{k}={v[0]}" for k, v in new_params.items()])
                        new_url = f"{base_parts.scheme}://{base_parts.netloc}{base_parts.path}?{new_query}"

                        if new_url not in page_urls:
                            page_urls.append(new_url)
                            print(f"Added page {page_num}: {new_url}")

                break

        # Method 3: Try to detect total results and calculate pages
        if len(page_urls) == 1:
            # Look for results count
            results_text = soup.get_text()

            # Look for patterns like "X wyników" or "X ogłoszeń"
            results_patterns = [
                r'(\d+)\s+wyników',
                r'(\d+)\s+ogłoszeń',
                r'(\d+)\s+results',
                r'(\d+)\s+listings'
            ]

            total_results = None
            for pattern in results_patterns:
                match = re.search(pattern, results_text, re.IGNORECASE)
                if match:
                    total_results = int(match.group(1))
                    break

            if total_results:
                # Assume 72 listings per page (from the URL limit parameter)
                listings_per_page = 72
                total_pages = (total_results + listings_per_page - 1) // listings_per_page

                print(f"Calculated from results count: {total_results} results, {total_pages} pages")

                if total_pages > 1:
                    base_parts = urlparse(base_search_url)
                    base_params = parse_qs(base_parts.query)

                    for page_num in range(2, min(total_pages + 1, 51)):  # Limit to 50 pages max
                        new_params = base_params.copy()
                        new_params['page'] = [str(page_num)]

                        new_query = '&'.join([f"{k}={v[0]}" for k, v in new_params.items()])
                        new_url = f"{base_parts.scheme}://{base_parts.netloc}{base_parts.path}?{new_query}"

                        page_urls.append(new_url)
                        print(f"Added calculated page {page_num}: {new_url}")

        print(f"Total pages to scrape: {len(page_urls)}")
        return page_urls

    def verify_existing_listings(self, existing_df, listing_ids_to_check):
        """Verify specific existing listings by checking for unavailable message"""
        print(f"Verifying {len(listing_ids_to_check)} existing listings for unavailable message...")

        removed_ids = set()
        still_available_ids = set()

        # Get the subset of dataframe for listings to check
        listings_to_check = existing_df[existing_df['listing_id'].isin(listing_ids_to_check)]

        for idx, row in listings_to_check.iterrows():
            listing_id = row['listing_id']
            url = row['url']

            print(f"Checking {listing_id}...")

            if self.check_listing_availability(url):
                # Listing is still available
                still_available_ids.add(listing_id)
                print(f"  -> {listing_id} is still AVAILABLE")
            else:
                # Listing shows unavailable message
                removed_ids.add(listing_id)
                print(f"  -> {listing_id} shows UNAVAILABLE message - marked for REMOVAL")

            time.sleep(2)  # Be respectful to the server

        print(f"\nVerification results:")
        print(f"- Still available: {len(still_available_ids)}")
        print(f"- Shows unavailable message: {len(removed_ids)}")

        return removed_ids, still_available_ids

    def check_listing_availability(self, url):
        """Check if a listing is still available by looking for the Polish 'unavailable' message"""
        soup = self.get_page_content(url)
        if not soup:
            print(f"  -> Could not fetch page content")
            return False

        # Look for the specific Polish "unavailable" messages
        unavailable_messages = [
            "To ogłoszenie jest już niedostępne",
            "Możliwe, że nieruchomość ma już nowego właściciela lub najemcę",
            "Poniżej znajdziesz podobne nieruchomości, które znaleźliśmy",
            "This listing is no longer available"
        ]

        page_text = soup.get_text()

        # Check for unavailable messages
        for message in unavailable_messages:
            if message in page_text:
                print(f"  -> Found unavailable message: '{message[:40]}...'")
                return False

        # Additional check: Look for specific CSS classes that indicate unavailability
        unavailable_selectors = [
            '.unavailable-listing',
            '.listing-removed',
            '.listing-not-found',
            '[data-testid="unavailable-listing"]',
            '.error-message',
            '.not-found-message',
            '.similar-listings',  # Often shown when original is unavailable
            '.recommended-listings'
        ]

        for selector in unavailable_selectors:
            elements = soup.select(selector)
            if elements:
                print(f"  -> Found unavailable indicator: selector '{selector}'")
                return False

        # Check if the page shows similar/recommended listings instead of the actual listing
        # This is a common pattern when a listing is unavailable
        similar_indicators = [
            "podobne nieruchomości",
            "similar properties",
            "recommended listings",
            "polecane oferty"
        ]

        for indicator in similar_indicators:
            if indicator.lower() in page_text.lower():
                # Check if this is the main content (not just a sidebar)
                main_content = soup.find('main') or soup.find('body')
                if main_content and indicator.lower() in main_content.get_text().lower():
                    print(f"  -> Page shows similar listings instead of original: '{indicator}'")
                    return False

        # If we can extract proper listing data, it's likely still available
        json_data = self.extract_json_data(soup)
        if json_data:
            try:
                ad_data = json_data['props']['pageProps']['ad']
                if ad_data and ad_data.get('id'):
                    print(f"  -> Listing data found - appears AVAILABLE")
                    return True
            except:
                pass

        # Check for proper listing title - if there's a detailed title, listing is likely available
        title_elem = soup.find('h1')
        if title_elem and title_elem.text.strip():
            title_text = title_elem.text.strip()
            # Check if title indicates unavailability
            unavailable_title_indicators = [
                "niedostępne",
                "unavailable",
                "not found",
                "nie znaleziono"
            ]

            if any(indicator in title_text.lower() for indicator in unavailable_title_indicators):
                print(f"  -> Title indicates unavailability: '{title_text[:50]}...'")
                return False
            elif len(title_text) > 10:
                print(f"  -> Found proper title - appears AVAILABLE")
                return True

        # Check for price information - if price is shown, listing is likely available
        price_selectors = [
            '[data-cy="adPageHeaderPrice"]',
            '.price',
            '[class*="price"]',
            '[data-testid*="price"]'
        ]

        for selector in price_selectors:
            price_elem = soup.select_one(selector)
            if price_elem and price_elem.get_text().strip():
                price_text = price_elem.get_text().strip()
                # Look for actual price numbers
                if re.search(r'\d+', price_text):
                    print(f"  -> Found price information - appears AVAILABLE")
                    return True

        print(f"  -> Could not confirm availability - treating as UNAVAILABLE")
        return False

    def scrape_all_listings(self):
        """Main method to scrape all listings with proper message-based removal"""
        base_url = "https://www.otodom.pl/pl/wyniki/wynajem/mieszkanie/mazowieckie/warszawa/warszawa/warszawa?limit=72&by=DEFAULT&direction=DESC"
        print("Loading existing data...")
        existing_df = self.load_existing_data()
        existing_ids = set(existing_df['listing_id'].tolist()) if not existing_df.empty else set()
        print(f"Found {len(existing_ids)} existing listings in Excel")

        print("Getting all page URLs...")
        page_urls = self.get_all_pages(base_url)
        print(f"Found {len(page_urls)} pages to scrape")

        # STEP 1: Collect ALL listing IDs currently on the website and mark existing ones as active
        print("\n" + "=" * 60)
        print("STEP 1: Processing website listings and marking active ones...")
        print("=" * 60)

        all_website_ids = set()
        all_website_urls = {}
        verified_active_ids = set()  # Listings found on website that exist in Excel

        for page_num, page_url in enumerate(page_urls, 1):
            print(f"Scanning page {page_num}/{len(page_urls)}")
            listing_urls = self.extract_listing_urls(page_url)

            if not listing_urls:
                print(f"No listings found on page {page_num}, stopping pagination")
                break

            for listing_url in listing_urls:
                listing_id = self.extract_listing_id(listing_url)
                if listing_id:
                    all_website_ids.add(listing_id)
                    all_website_urls[listing_id] = listing_url

                    # If this listing exists in Excel, mark it as verified active
                    if listing_id in existing_ids:
                        verified_active_ids.add(listing_id)
                        print(f"  -> {listing_id} marked as ACTIVE (found on website)")

            time.sleep(1)

        print(f"Found {len(all_website_ids)} total listings on website")
        print(f"Marked {len(verified_active_ids)} existing listings as active")

        # STEP 2: Extract data for new listings
        print("\n" + "=" * 60)
        print("STEP 2: Processing new listings...")
        print("=" * 60)

        new_listing_ids = all_website_ids - existing_ids
        new_listings = []
        skipped_count = 0

        if new_listing_ids:
            print(f"Extracting data for {len(new_listing_ids)} new listings...")

            for listing_id in new_listing_ids:
                listing_url = all_website_urls[listing_id]
                print(f"Processing new listing: {listing_id}")

                listing_data = self.extract_listing_data(listing_url)

                if listing_data == "SKIP":
                    skipped_count += 1
                    print(f"  -> Skipped {listing_id} (no street address)")
                elif listing_data:
                    new_listings.append(listing_data)
                    print(f"  -> Successfully extracted {listing_id}")
                else:
                    print(f"  -> Failed to extract data for {listing_id}")

                time.sleep(1)

        # STEP 3: Check unmarked listings for unavailable message
        print("\n" + "=" * 60)
        print("STEP 3: Checking unmarked listings for removal...")
        print("=" * 60)

        unmarked_ids = existing_ids - verified_active_ids
        removed_ids = set()

        if unmarked_ids:
            print(f"Found {len(unmarked_ids)} existing listings not marked as active")
            print("Checking these listings for unavailable message...")

            for listing_id in unmarked_ids:
                # Get URL for this listing from existing data
                listing_row = existing_df[existing_df['listing_id'] == listing_id]
                if not listing_row.empty:
                    url = listing_row['url'].iloc[0]
                    print(f"Checking {listing_id}...")

                    if not self.check_listing_availability(url):
                        # Shows unavailable message - mark for removal
                        removed_ids.add(listing_id)
                        print(f"  -> {listing_id} shows UNAVAILABLE message - will be REMOVED")
                    else:
                        # Still available - keep in Excel
                        print(f"  -> {listing_id} is still available - will be KEPT")

                    time.sleep(2)  # Be respectful to server
        else:
            print("All existing listings were marked as active - no need to check for removal")

        # STEP 4: Remove listings that show unavailable message
        print("\n" + "=" * 60)
        print("STEP 4: Removing listings with unavailable message...")
        print("=" * 60)

        if removed_ids:
            print(f"Removing {len(removed_ids)} listings that show unavailable message:")
            for listing_id in list(removed_ids)[:10]:
                print(f"  - {listing_id}")
            if len(removed_ids) > 10:
                print(f"  ... and {len(removed_ids) - 10} more")

            # Actually remove from dataframe
            if not existing_df.empty:
                original_count = len(existing_df)
                existing_df = existing_df[~existing_df['listing_id'].isin(removed_ids)]
                removed_count = original_count - len(existing_df)
                print(f"Successfully removed {removed_count} listings from Excel data")

            self.existing_df_updated = existing_df
        else:
            print("No listings show unavailable message - no removals needed")
            self.existing_df_updated = existing_df

        # Store data for save_data method
        self.all_listings = new_listings
        self.verified_active_ids = verified_active_ids

        # Save updated data
        self.save_data()

        print("\n" + "=" * 60)
        print("SCRAPING COMPLETED - FINAL SUMMARY:")
        print("=" * 60)
        print(f"- Existing listings marked as active: {len(verified_active_ids)}")
        print(f"- Existing listings kept (not marked but still available): {len(unmarked_ids - removed_ids)}")
        print(f"- New listings added: {len(new_listings)}")
        print(f"- Listings removed (unavailable message): {len(removed_ids)}")
        print(f"- Skipped new listings (no address): {skipped_count}")

        total_final = len(verified_active_ids) + len(unmarked_ids - removed_ids) + len(new_listings)
        print(f"- Total listings in final Excel: {total_final}")

        return pd.DataFrame(self.all_listings) if self.all_listings else pd.DataFrame()


# Usage
if __name__ == "__main__":
    scraper = OtodomScraper()
    df = scraper.scrape_all_listings()
    print(f"Final dataset contains {len(df)} new listings")
