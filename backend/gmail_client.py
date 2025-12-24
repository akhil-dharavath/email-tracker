import os.path
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import datetime

from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailClient:
    def __init__(self, client_config=None, token_data=None, credentials_path=None, token_path=None):
        self.client_config = client_config
        self.token_data = token_data
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None

    def authenticate(self):
        creds = None
        
        # 1. Try loading from dictionary (Env Var source)
        if self.token_data:
            try:
                creds = Credentials.from_authorized_user_info(self.token_data, SCOPES)
            except Exception:
                creds = None

        # 2. Try loading from file
        if not creds and self.token_path and os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            except Exception:
                creds = None

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Token refresh failed: {e}")
                    creds = None
            
            if not creds:
                # Flow from Config Dict (Env) or File
                flow = None
                if self.client_config:
                    flow = InstalledAppFlow.from_client_config(self.client_config, SCOPES)
                elif self.credentials_path and os.path.exists(self.credentials_path):
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                else:
                     raise FileNotFoundError(f"Could not find credentials in Env or at {self.credentials_path}")
                
                try:
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    raise Exception(f"Authentication failed. Details: {e}")
            
            # Save the credentials for the next run (file persistence if path provided)
            # We also might want to print the JSON for the user to put in ENV if no file path
            if self.token_path:
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())
            else:
                 print("\n[INFO] Generated Token JSON (Add this to your .env as GOOGLE_TOKEN_JSON):")
                 print(creds.to_json())

        try:
            self.service = build('gmail', 'v1', credentials=creds)
        except HttpError as e:
             raise Exception(f"Failed to build Gmail service: {e}")

    def fetch_emails(self, max_results=10):
        if not self.service:
            self.authenticate()

        try:
            results = self.service.users().messages().list(userId='me', maxResults=max_results).execute()
        except HttpError as e:
            if e.resp.status == 403:
                raise Exception("403 Forbidden: Enable Gmail API in Cloud Console AND add your email to 'Test Users' in OAuth Consent Screen.")
            raise e
            
        messages = results.get('messages', [])
        
        email_list = []
        for msg in messages:
            try:
                full_msg = self.service.users().messages().get(userId='me', id=msg['id']).execute()
                payload = full_msg['payload']
                headers = payload.get('headers', [])
                
                subject = ""
                sender = ""
                date_str = ""
                
                for h in headers:
                    name = h['name'].lower()
                    if name == 'subject': subject = h['value']
                    if name == 'from': sender = h['value']
                    if name == 'date': date_str = h['value']

                # Get Body
                body = ""
                if 'parts' in payload:
                    for part in payload['parts']:
                        if part['mimeType'] == 'text/plain':
                            data = part['body'].get('data')
                            if data:
                                body += base64.urlsafe_b64decode(data).decode()
                                break
                elif 'body' in payload:
                    data = payload['body'].get('data')
                    if data:
                        body += base64.urlsafe_b64decode(data).decode()
                
                # Cleanup Body
                if not body:
                    body = "(No content)"
                else:
                     # simplistic strip of html if any leaks through or just plain text cleanup
                     # For more robust HTML handling, use BeautifulSoup
                     soup = BeautifulSoup(body, 'html.parser')
                     body = soup.get_text()

                # Parse Date
                timestamp = datetime.datetime.now().isoformat()
                if date_str:
                    try:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(date_str)
                        timestamp = dt.isoformat()
                    except Exception:
                        pass # Fallback to now
                
                # Check Labels for UNREAD
                is_unread = 'UNREAD' in full_msg.get('labelIds', [])

                email_list.append({
                    "id": msg['id'],
                    "subject": subject,
                    "body": body[:500], # Truce for analysis
                    "sender": sender,
                    "timestamp": timestamp,
                    "isUnread": is_unread
                })
            except Exception as e:
                print(f"Error parsing email {msg['id']}: {e}")
                continue
                
        return email_list
