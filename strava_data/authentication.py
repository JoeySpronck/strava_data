import json
from stravalib.client import Client
import webbrowser
import os
os.environ["SILENCE_TOKEN_WARNINGS"] = "true"

# Got main logic from:
# https://github.com/stravalib/stravalib/blob/main/docs/get-started/how-to-get-strava-data-python.md
def login():
    with open("secrets/client_secrets.txt", "r") as f:
        # This file should contain your client_id and client_secret, separated by a comma
        client_id, client_secret = f.read().strip().split(",")
    client = Client()

    if not os.path.exists("secrets/strava_token.json"):
        request_scope = ["read_all", "profile:read_all", "activity:read_all"]
        redirect_url = "http://127.0.0.1:5000/authorization"
        url = client.authorization_url(
            client_id=client_id,
            redirect_uri=redirect_url,
            scope=request_scope,
        )
        webbrowser.open(url)
        print("""You will see a url that looks like this. """,
            """http://127.0.0.1:5000/authorization?state=&code=12323423423423423423423550&scope=read,activity:read_all,profile:read_all,read_all")""",
            """Copy the values between code= and & in the url that you see in the browser. """)
        code = input("Please enter the code that you received: ")
        token_response = client.exchange_code_for_token(
            client_id=client_id, client_secret=client_secret, code=code)
        with open("secrets/strava_token.json", "w") as f:
            json.dump(token_response, f)
    else:
        print("You have already authenticated once before. Refreshing your token now.")
        with open("secrets/strava_token.json") as f:
            token_response = json.load(f)
        refresh_response = client.refresh_access_token(
            client_id=client_id,  # Stored in the secrets.txt file above
            client_secret=client_secret,
            refresh_token=token_response["refresh_token"],  # Stored in your JSON file
        )
    # Check that the refresh worked
    athlete = client.get_athlete()
    print(f"Hi {athlete.firstname}, authentication successful!")
    return client
