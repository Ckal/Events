import gradio as gr
import requests
from bs4 import BeautifulSoup
import pytz
from datetime import datetime, timedelta
import logging
import traceback
from typing import List, Dict, Any
import hashlib
import icalendar
import uuid
import re
import json
import os

# Hugging Face imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True # TODO change back to true to use local llm 
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Hugging Face Inference Client
from huggingface_hub import InferenceClient

class EventScraper:
    def __init__(self, urls, timezone='Europe/Berlin'):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Timezone setup
        self.timezone = pytz.timezone(timezone)
        
        # URLs to scrape
        self.urls = urls if isinstance(urls, list) else [urls]
        
        # Event cache to prevent duplicates
        self.event_cache = set()
        
        # iCal calendar
        self.calendar = icalendar.Calendar()
        self.calendar.add('prodid', '-//Event Scraper//example.com//')
        self.calendar.add('version', '2.0')
        
        # Model and tokenizer will be loaded on first use
        self.model = None
        self.tokenizer = None
        self.client = None
    
    def setup_llm(self):
        """Setup Hugging Face LLM for event extraction"""
        # Try local model first
        if TRANSFORMERS_AVAILABLE:
            try:
                model_name = "meta-llama/Llama-3.2-1B-Instruct" # 3B is very slow on HF :(
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    return_dict_in_generate=False,
                    device_map='auto'
                )
                return
            except Exception as local_err:
                gr.Warning(f"Local model setup failed: {str(local_err)}")
        
        # Fallback to Inference Client
        try:
            # Try to get Hugging Face token from environment
            hf_token = os.getenv('HF_TOKEN')
            
            # Setup Inference Client
            if hf_token:
                self.client = InferenceClient(
                    model="meta-llama/Llama-3.2-3B-Instruct", 
                    token=hf_token 
                )
            else:
                # Public model access without token
                self.client = InferenceClient(
                    model="meta-llama/Llama-3.2-3B-Instruct" 
                )
        except Exception as e:
            gr.Warning(f"Inference Client setup error: {str(e)}")
            raise
    
    def generate_with_model(self, prompt):
        """Generate text using either local model or inference client"""
        print("------ PROMPT ------------")
        print(prompt)
        print("------ PROMPT ------------")
        if self.model and self.tokenizer:
            # Use local model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs.input_ids, 
                max_new_tokens=12000, 
                do_sample=True, 
                temperature=0.9
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        elif self.client:
            # Use Inference Client
            return self.client.text_generation(
                prompt, 
                max_new_tokens=2000, 
                temperature=0.9
            )
        
        else:
            raise ValueError("No model or client available for text generation")
    
    def fetch_webpage_content(self, url):
        """Fetch webpage content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            gr.Warning(f"Error fetching {url}: {str(e)}")
            return ""
    
    def extract_text_from_html(self, html_content):
        """Extract readable text from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split()[:2000])
    
    def generate_event_extraction_prompt(self, text):
        """Create prompt for LLM to extract event details"""

        prompt=f'''
        <|start_header_id|>system<|end_header_id|>
        
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        You are an event extraction assistant. 
        Find and extract all events from the following text. 
        For each event, provide:
        - Exact event name
        - Date (DD.MM.YYYY)
        - Time (HH:MM if available)
        - Location
        - Short description

        Important: Extract ALL possible events. 
        Text to analyze:
        {text}

        Output ONLY a JSON list of events like this - Response Format:
        [
          {{
            "name": "Event Name",
            "date": "07.12.2024",
            "time": "19:00",
            "location": "Event Location",
            "description": "Event details"
          }}
        ]

        If NO events are found, return an empty list [].
        Only return the json. nothing else. no comments.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
 
        return prompt
    
    def parse_llm_response(self, response):
        """Parse LLM's text response into structured events"""
        try:
            # Clean the response and handle nested lists
            response = response.strip()
            
            # Try parsing as JSON, handling potential nested structures
            def flatten_events(data):
                if isinstance(data, list):
                    flattened = []
                    for item in data:
                        if isinstance(item, list):
                            flattened.extend(flatten_events(item))
                        elif isinstance(item, dict):
                            flattened.append(item)
                    return flattened
                return []
    
            try:
                # First, attempt direct JSON parsing
                events = json.loads(response)
                events = flatten_events(events)
            except json.JSONDecodeError:
                # If direct parsing fails, try extracting JSON
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL | re.MULTILINE)
                if json_match:
                    try:
                        events = json.loads(json_match.group(0))
                        events = flatten_events(events)
                    except json.JSONDecodeError:
                        events = []
                else:
                    events = []
            
            # Clean and validate events
            cleaned_events = []
            for event in events:
                # Ensure each event has at least a name
                if event.get('name'):
                    # Set default values if missing
                    event.setdefault('date', '')
                    event.setdefault('time', '')
                    event.setdefault('location', '')
                    event.setdefault('description', '')
                    cleaned_events.append(event)
            
            return cleaned_events
        
        except Exception as e:
            gr.Warning(f"Parsing error: {str(e)}")
            return []
    
    def scrape_events(self):
        """Main method to scrape events from all URLs"""
        # Ensure LLM is set up
        self.setup_llm()
        
        all_events = []
        
        for url in self.urls:
            try:
                # Fetch webpage
                html_content = self.fetch_webpage_content(url)
                
                # Extract readable text
                text_content = self.extract_text_from_html(html_content)
                
                # Generate prompt
                prompt = self.generate_event_extraction_prompt(text_content)
                
                # Generate response
                response = self.generate_with_model(prompt)

                print("------ response ------------")
                print(response)
                print("------ response ------------")
                
                # Parse events
                parsed_events = self.parse_llm_response(response)
                
                # Deduplicate and add
                for event in parsed_events:
                    event_hash = hashlib.md5(str(event).encode()).hexdigest()
                    if event_hash not in self.event_cache:
                        self.event_cache.add(event_hash)
                        all_events.append(event)
                        
                        # Create and add iCal event
                        try:
                            ical_event = self.create_ical_event(event)
                            self.calendar.add_component(ical_event)
                        except Exception as ical_error:
                            gr.Warning(f"iCal creation error: {str(ical_error)}")
                
            except Exception as e:
                gr.Warning(f"Error processing {url}: {str(e)}")
        
        return all_events
    
def create_ical_event(self, event):
    """Convert event to iCal format"""
    ical_event = icalendar.Event()
    
    # Set unique identifier
    ical_event.add('uid', str(uuid.uuid4()))
    
    # Add summary (name)
    ical_event.add('summary', event.get('name', 'Unnamed Event'))
    
    # Add description
    ical_event.add('description', event.get('description', ''))
    
    # Add location
    if event.get('location'):
        ical_event.add('location', event['location'])
    
    # Handle date and time
    try:
        # Parse date
        if event.get('date'):
            try:
                event_date = datetime.strptime(event['date'], '%d.%m.%Y').date()
                
                # Parse time if available
                event_time = datetime.strptime(event.get('time', '00:00'), '%H:%M').time() if event.get('time') else datetime.min.time()
                
                # Combine date and time
                event_datetime = datetime.combine(event_date, event_time)
                
                # Localize the datetime to the specified timezone
                localized_datetime = self.timezone.localize(event_datetime)
                
                # For all-day events, set to start at midnight and end just before midnight the next day
                if event_time == datetime.min.time():
                    start_datetime = localized_datetime.replace(hour=0, minute=0, second=0)
                    end_datetime = (start_datetime + timedelta(days=1)).replace(hour=23, minute=59, second=59)
                    
                    # Add properties for all-day event
                    ical_event.add('dtstart', start_datetime.date())
                    ical_event.add('dtend', end_datetime.date())
                    ical_event.add('x-microsoft-cdo-alldayevent', 'TRUE')
                else:
                    # For events with specific time, set 1-hour duration if not specified
                    end_datetime = localized_datetime + timedelta(hours=1)
                    
                    # Use TZID format
                    ical_event['dtstart'] = icalendar.prop.vDDDTypes(localized_datetime)
                    ical_event['dtstart'].params['TZID'] = 'Europe/Berlin'
                    
                    ical_event['dtend'] = icalendar.prop.vDDDTypes(end_datetime)
                    ical_event['dtend'].params['TZID'] = 'Europe/Berlin'
                
            except ValueError as date_err:
                gr.Warning(f"Date parsing error: {date_err}")
        
    except Exception as e:
        gr.Warning(f"iCal event creation error: {str(e)}")
    
    return ical_event
    
    
    def get_ical_string(self):
        """Return iCal as a string"""
        return self.calendar.to_ical().decode('utf-8')

def scrape_events_with_urls(urls):
    """Wrapper function for Gradio interface"""
    # Split URLs by newline or comma
    url_list = [url.strip() for url in re.split(r'[\n,]+', urls) if url.strip()]
    
    if not url_list:
        gr.Warning("Please provide at least one valid URL.")
        return "", ""
    
    try:
        # Initialize scraper
        scraper = EventScraper(url_list)
        
        # Scrape events
        events = scraper.scrape_events()
        
        # Prepare events output
        events_str = json.dumps(events, indent=2)
        
        # Get iCal string
        ical_string = scraper.get_ical_string()
        
        return events_str, ical_string
    
    except Exception as e:
        gr.Warning(f"Error in event scraping: {str(e)}")
        return "", ""

# Create Gradio Interface
def create_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Event Scraper 🗓️")
        gr.Markdown("Scrape events from web pages using an AI-powered event extraction tool.")
        
        with gr.Row():
            with gr.Column():
                url_input = gr.Textbox(
                    label="Enter URLs (comma or newline separated)", 
                    placeholder="https://example.com/events\nhttps://another-site.com/calendar"
                )
                scrape_btn = gr.Button("Scrape Events", variant="primary")
        
        with gr.Row():
            with gr.Column():
                events_output = gr.Textbox(label="Extracted Events (JSON)", lines=10)
            with gr.Column():
                ical_output = gr.Textbox(label="iCal Export", lines=10)
        
        scrape_btn.click(
            fn=scrape_events_with_urls, 
            inputs=url_input, 
            outputs=[events_output, ical_output]
        )
        
        gr.Markdown("**Note:** Requires an internet connection and may take a few minutes to process.")
        gr.Markdown("Set HF_TOKEN environment variable for authenticated access.")
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch()