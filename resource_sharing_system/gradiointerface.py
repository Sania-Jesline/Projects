import gradio as gr
import requests
import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from modeltry import semantic_search_from_firebase
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate('C:/Users/sania/Documents/Projects/Mini Project/trying/gradio-auth-app-firebase-adminsdk-fbsvc-ace9052cd4.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://gradio-auth-app-default-rtdb.asia-southeast1.firebasedatabase.app"
  # Use your Firebase Realtime Database URL
})


# Firebase config
FIREBASE_API_KEY = "AIzaSyCH30eqZfA0cVLPIAfenJ--f7RJ9fP6bZI"
DATABASE_URL = "https://gradio-auth-app-default-rtdb.asia-southeast1.firebasedatabase.app"

# Global session
session = {
    "uid": None,
    "id_token": None,
    "username": None,
    "email": None
}

# Firebase Authentication functions
def sign_in(email, password):
    # Basic input validation
    if "@" not in email or "." not in email.split("@")[-1]:
        return "Login failed: Invalid email format"

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
        data = res.json()

        # ✅ Store both UID keys
        session["uid"] = data["localId"]
        session["local_id"] = data["localId"]  # 🔧 Fix for chat + other functions
        session["id_token"] = data["idToken"]

        user_info = fetch_user_info()
        user_info_json = json.loads(user_info)
        session["username"] = user_info_json.get("username", "")
        session["email"] = user_info_json.get("email", "")
        
        return "Login successful"
    
    except requests.exceptions.HTTPError as e:
        error_message = e.response.json().get('error', {}).get('message', 'Unknown error')
        return f"Login failed: {error_message}"

def sign_up(email, username, password, confirm_password):
    if password != confirm_password:
        return "Passwords do not match"
    
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    try:
        # Sign up the user
        res = requests.post(url, json=payload)
        res.raise_for_status()
        data = res.json()

        session["uid"] = data["localId"]
        session["id_token"] = data["idToken"]
        session["email"] = email
        session["username"] = username
        
        # Store user info in Firebase under "users" collection
        user_data = {
            "email": email,
            "username": username
        }
        store_user_info_in_firebase(user_data)
        
        return "Signup successful! Please log in."
    except requests.exceptions.HTTPError as e:
        return f"Signup failed: {e.response.json().get('error', {}).get('message', 'Unknown error')}"

def store_user_info_in_firebase(user_data):
    uid = session.get("uid")
    id_token = session.get("id_token")
    if not uid or not id_token:
        return "Error: Not logged in"
    
    try:
        url = f"{DATABASE_URL}/users/{uid}.json?auth={id_token}"
        res = requests.put(url, json=user_data)
        res.raise_for_status()
    except Exception as e:
        return f"Error storing user info: {e}"

def fetch_user_info():
    uid = session.get("uid")
    id_token = session.get("id_token")
    if not uid or not id_token:
        return None
    
    try:
        url = f"{DATABASE_URL}/users/{uid}.json?auth={id_token}"
        res = requests.get(url)
        res.raise_for_status()
        return res.text
    except Exception as e:
        return None

# Resource Matching functions
class ResourceMatcher:
    def __init__(self):
        print("Initializing BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.resources = []
        print("Model initialized!")
        
    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()
    
    def add_resource(self, type, name, description, owner, status, contact):
        resource_info = {
            'type': type,
            'name': name,
            'description': description,
            'owner': owner,
            'status': status,
            'contact': contact
        }
        description_text = f"{type} {name} {description}"
        embedding = self.get_bert_embedding(description_text)
        resource_info['embedding'] = embedding
        self.resources.append(resource_info)
        return "Resource added successfully!"
    
    def search_resources(self, query):
        if not self.resources:
            return "No resources available in the system."
        query_embedding = self.get_bert_embedding(query)
        matches = []
        for resource in self.resources:
            similarity = cosine_similarity(query_embedding, resource['embedding'])[0][0]
            if similarity > 0.5:  # Threshold for matching
                matches.append({
                    'resource': resource,
                    'similarity': similarity
                })
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        if not matches:
            return "No matching resources found."
        results = []
        for idx, match in enumerate(matches, 1):
            resource = match['resource']
            result = f"""
Match #{idx} (Similarity: {match['similarity']:.2f})
Type: {resource['type']}
Name: {resource['name']}
Description: {resource['description']}
Owner: {resource['owner']}
Status: {resource['status']}
Contact: {resource['contact']}
-------------------"""
            results.append(result)
        return "\n".join(results)

matcher = ResourceMatcher()

# Gradio Interface
def add_resource_interface(type, name, description, owner, status, contact):
    try:
        return matcher.add_resource(type, name, description, owner, status, contact)
    except Exception as e:
        return f"Error: {str(e)}"

def start_chatroom(resource_id, owner_id):
    requester_id = session.get("local_id")
    
    if not requester_id or not owner_id:
        return "Missing user info"

    user_pair = sorted([requester_id, owner_id])
    chat_id = f"{user_pair[0]}_{user_pair[1]}"

    # Check if chat exists; create if not
    chat_url = f"{DATABASE_URL}/chats/{chat_id}.json?auth={session['id_token']}"
    res = requests.get(chat_url)
    if res.status_code == 200 and res.json() is None:
        requests.put(chat_url, json={"messages": []})

    session["current_chat_id"] = chat_id
    return get_messages()

# Sample search resources function returning resources


def fetch_profile_card():
    email = session.get("email", "N/A")
    username = session.get("username", "N/A")
    return f"""
        <div style='border: 1px solid #444; padding: 16px; border-radius: 8px; background: #2c2f3e; color: rgba(255, 255, 255, 0.9); margin-bottom: 16px;'>
            <h2>Profile</h2>
            <p><strong>Email:</strong> {email}</p>
            <p><strong>Username:</strong> {username}</p>
        </div>
    """

def fetch_user_resources_html():
    uid = session.get("uid")
    id_token = session.get("id_token")
    username = session.get("username")
    if not uid or not id_token or not username:
        return "<p style='color: red;'>Error: Not logged in</p>"

    try:
        url = f"{DATABASE_URL}/resources.json?auth={id_token}"
        res = requests.get(url)
        res.raise_for_status()
        all_resources = res.json()

        user_resources = []
        for user_id, resources in all_resources.items():
            for resource_id, resource in resources.items():
                if resource.get("owner") == username:
                    filtered = resource.copy()
                    filtered["resource_id"] = resource_id
                    user_resources.append(filtered)

        if not user_resources:
            return "<p>No resources found for this user.</p>"

        html = "<h2 style='color: white;'>Resources</h2><div style='display: flex; flex-wrap: wrap; gap: 16px;'>"
        for r in user_resources:
            html += f"""
                <div style='border: 1px solid #444; padding: 16px; border-radius: 8px; width: 500px; background: #2c2f3e; color: rgba(255, 255, 255, 0.9);'>
                    <h3>{r.get('name', 'No Name')}</h3>
                    <p><strong>Type:</strong> {r.get('type', 'N/A')}</p>
                    <p><strong>Description:</strong> {r.get('description', 'N/A')}</p>
                    <p><strong>Contact:</strong> {r.get('contact', 'N/A')}</p>
                    <p><strong>Availability:</strong> {r.get('availability', 'N/A')}</p>
                </div>
            """
        html += "</div>"
        return html
    except Exception as e:
        return f"<p style='color: red;'>Error fetching resources: {e}</p>"
    
def get_resource_toggle_list():
    uid = session.get("uid")
    id_token = session.get("id_token")
    username = session.get("username")
    if not uid or not id_token or not username:
        return [], []

    try:
        url = f"{DATABASE_URL}/resources.json?auth={id_token}"
        res = requests.get(url)
        res.raise_for_status()
        all_resources = res.json()

        toggles = []
        labels = []
        for user_id, resources in all_resources.items():
            for resource_id, resource in resources.items():
                if resource.get("owner") == username:
                    key = f"{user_id}|{resource_id}"
                    toggle = gr.Checkbox(label=resource["name"], value=(resource.get("availability") == "Available"))
                    toggles.append((key, toggle))
                    labels.append(toggle)

        return toggles, labels
    except:
        return [], []

def save_resource_changes_from_toggles(*toggle_values):
    uid = session.get("uid")
    id_token = session.get("id_token")
    if not uid or not id_token:
        return "Error: Not logged in"

    try:
        url = f"{DATABASE_URL}/resources.json?auth={id_token}"
        res = requests.get(url)
        res.raise_for_status()
        all_resources = res.json()

        flat_resources = []
        for user_id, resources in all_resources.items():
            for resource_id, resource in resources.items():
                if resource.get("owner") == session["username"]:
                    flat_resources.append((user_id, resource_id, resource.get("name")))

        for idx, (user_id, resource_id, name) in enumerate(flat_resources):
            availability_value = "Available" if toggle_values[idx] else "Unavailable"
            update_url = f"{DATABASE_URL}/resources/{user_id}/{resource_id}.json?auth={id_token}"
            requests.patch(update_url, json={"availability": availability_value})

        return "All availability values updated."
    except Exception as e:
        return f"Error saving changes: {e}"

with gr.Blocks() as demo:
    chat_mode = gr.State(False)
    user_state = gr.State()

    gr.Markdown("# Resource Sharing Platform")

    # ----- Signup -----
    with gr.Tab("Signup"):
        email_input = gr.Textbox(label="Email", placeholder="Enter your email")
        username_input = gr.Textbox(label="Username", placeholder="Enter your username")
        password_input = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
        confirm_password_input = gr.Textbox(label="Confirm Password", type="password", placeholder="Re-enter your password")
        signup_button = gr.Button("Sign Up")
        signup_output = gr.Textbox(label="Signup Output")
        
        signup_button.click(sign_up, inputs=[email_input, username_input, password_input, confirm_password_input], outputs=signup_output)

    


    # Login Tab
    with gr.Tab("Login"):
        email_input_login = gr.Textbox(label="Email")
        password_input_login = gr.Textbox(label="Password", type="password")
        login_button = gr.Button("Login")
        login_output = gr.Textbox(label="Login Output")
        login_button.click(sign_in, inputs=[email_input_login, password_input_login], outputs=login_output)

    

    # ----- Profile -----
    with gr.Tab("Profile"):
        profile_output = gr.HTML(label="Profile Info")
        resources_output = gr.HTML(label="My Resources")
        refresh_button = gr.Button("Refresh My Resources")

        def refresh_all():
            return fetch_profile_card(), fetch_user_resources_html()

        refresh_button.click(refresh_all, outputs=[profile_output, resources_output])

    # ----- Add Resources -----
    with gr.Tab("Add Resources"):
        name_input = gr.Textbox(label="Resource Name")
        type_input = gr.Dropdown(choices=["notes", "book", "hardware", "other"], label="Resource Type")
        description_input = gr.Textbox(label="Description")
        availability_input = gr.Dropdown(choices=["available", "unavailable"], label="Availability")
        contact_input = gr.Textbox(label="Contact")
        status_input = gr.Dropdown(choices=["lending", "giveaway"], label="Status")
        add_resource_button = gr.Button("Add Resource")
        add_resource_output = gr.Textbox(label="Add Resource Output")

        def add_resource(name, type, description, availability, contact, status):
            uid = session.get("uid")
            id_token = session.get("id_token")
            username = session.get("username")
            if not uid or not id_token or not username:
                return "Error: Not logged in"
            resource_id = str(uuid.uuid4())
            resource_data = {
                "name": name, "type": type, "description": description,
                "availability": availability, "contact": contact,
                "status": status, "owner": username
            }
            try:
                url = f"{DATABASE_URL}/resources/{uid}/{resource_id}.json?auth={id_token}"
                requests.put(url, json=resource_data)
                return "Resource added successfully"
            except Exception as e:
                return f"Error adding resource: {e}"

        add_resource_button.click(add_resource, inputs=[name_input, type_input, description_input, availability_input, contact_input, status_input], outputs=add_resource_output)

    # ----- Search Resources -----
    with gr.Tab("Search Resources"):
        search_input = gr.Textbox(label="Search Query")
        search_btn = gr.Button("Search")
        search_output = gr.Markdown()
        chat_dropdown = gr.Dropdown(label="Select a resource to chat with the owner", interactive=True)
        start_chat_btn = gr.Button("Start Chat")
        chat_display = gr.HTML(visible=False)
        message_box = gr.Textbox(label="Type your message", visible=False)
        send_btn = gr.Button("Send", visible=False)

        def search_resources_interface(query):
            id_token = session.get("id_token")
            if not id_token:
                return "Not logged in", gr.update(choices=[], value=None)

            # Fetch resource data
            res = requests.get(f"{DATABASE_URL}/resources.json?auth={id_token}")
            firebase_data = res.json() or {}

            # Fetch user data for mapping owner names
            users_res = requests.get(f"{DATABASE_URL}/users.json?auth={id_token}")
            users_data = users_res.json() or {}

            from modeltry import semantic_search_from_firebase
            results = semantic_search_from_firebase(query, firebase_data)

            filtered = [r for r in results if r.get("score", 0) > 0.2]
            filtered.sort(key=lambda r: r["score"], reverse=True)

            if not filtered:
                return "No relevant resources found.", gr.update(choices=[], value=None)

            dropdown_choices = []
            display_texts = []

            for r in filtered:
                name = r.get('name', 'No Name')
                description = r.get('description', '')
                rid = r.get('resource_id')
                uid = r.get('user_id')
                owner_name = users_data.get(uid, {}).get("username", "Unknown")
                label = f"{name} - {description} (Owner: {owner_name})"
                value = f"{uid}|{rid}"
                dropdown_choices.append((label, value))
                display_texts.append(f"### {name}\n**Description:** {description}\n**Owner:** {owner_name}")

            return "\n\n".join(display_texts), gr.update(choices=dropdown_choices, value=None)

        def start_chatroom(resource_id, owner_id):
            requester_id = session.get("local_id")
            if not requester_id or not owner_id:
                return "Missing user info"

            chat_id = f"{'_'.join(sorted([requester_id, owner_id]))}"
            url = f"{DATABASE_URL}/chats/{chat_id}.json?auth={session['id_token']}"
            if not requests.get(url).json():
                requests.put(url, json={"messages": []})

            session["current_chat_id"] = chat_id
            return get_messages()

        def get_messages():
            chat_id = session.get("current_chat_id")
            if not chat_id:
                return "No chat selected."
            url = f"{DATABASE_URL}/chats/{chat_id}/messages.json?auth={session['id_token']}"
            messages = list((requests.get(url).json() or {}).values())
            html = '<div style="display:flex;flex-direction:column;gap:10px;padding:10px;">'
            for m in messages:
                is_sender = m.get("sender_id") == session.get("local_id")
                sender = "You" if is_sender else "Anonymous"
                align = "flex-end" if is_sender else "flex-start"
                bg = "#dcf8c6" if is_sender else "#fff"
                html += f"""
    <div style="display:flex;justify-content:{align};">
    <div style="background:{bg};padding:12px 16px;border-radius:18px;max-width:70%;box-shadow:0 2px 5px rgba(0,0,0,0.1);">
        <b style="font-size:12px;color:#2E1A17;">{sender}</b><br>
        <span style="color:#2E1A17;">{m.get('text', '')}</span>
    </div>
    </div>"""
            html += '</div>'
            return html

        def send_message(message):
            chat_id = session.get("current_chat_id")
            sender_id = session.get("local_id")
            if not chat_id or not sender_id or not message.strip():
                return get_messages()
            url = f"{DATABASE_URL}/chats/{chat_id}/messages.json?auth={session['id_token']}"
            requests.post(url, json={"sender_id": sender_id, "text": message.strip()})
            return get_messages()

        def handle_start_chat(selected_value):
            if not selected_value or "|" not in selected_value:
                return "Invalid selection", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            uid, rid = selected_value.split("|")
            chat_html = start_chatroom(rid, uid)
            return chat_html, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

        # Link button and components
        search_btn.click(
            search_resources_interface,
            inputs=[search_input],
            outputs=[search_output, chat_dropdown]
        )

        start_chat_btn.click(
            handle_start_chat,
            inputs=[chat_dropdown],
            outputs=[chat_display, message_box, send_btn, chat_display]
        )

        send_btn.click(
            send_message,
            inputs=[message_box],
            outputs=[chat_display]
        )


    # ----- My Chats -----
    
    with gr.Tab("My Chats"):
        my_chats_dropdown = gr.Dropdown(label="My Chats", interactive=True)
        refresh_chats_btn = gr.Button("Refresh My Chats")
        load_chat_btn = gr.Button("Load Chat")
        chat_display2 = gr.HTML(visible=False)
        message_box2 = gr.Textbox(label="Type your message", visible=False)
        send_btn2 = gr.Button("Send", visible=False)

        chat_id_map = {}  # To map user-friendly labels to chat IDs

        def get_messages_chat(chat_id, user_id):
            chat_ref = db.reference(f'chats/{chat_id}/messages')
            messages = chat_ref.get() or {}

            users_ref = db.reference('users')
            users = users_ref.get() or {}

            if not messages:
                return "<p>No messages found.</p>"

            html = """
            <div style='font-family: sans-serif; display: flex; flex-direction: column; gap: 10px;'>
            """

            for _, msg in sorted(messages.items()):
                sender_id = msg.get("sender_id")
                text = msg.get("text", "").replace("\n", "<br>")

                if not sender_id or not text:
                    continue

                sender_name = users.get(sender_id, {}).get("username", "Unknown")

                if sender_id == user_id:
                    # Sent message
                    html += f"""
                    <div style='align-self: flex-end; max-width: 60%; background-color: #dcfce7; color: #000;
                                padding: 10px 15px; border-radius: 20px; border-bottom-right-radius: 0;
                                font-size: 14px;'>
                        <div style='font-weight: bold; font-size: 12px; margin-bottom: 4px;'>You</div>
                        {text}
                    </div>
                    """
                else:
                    # Received message
                    html += f"""
                    <div style='align-self: flex-start; max-width: 60%; background-color: white; color: #000;
                                padding: 10px 15px; border-radius: 20px; border-bottom-left-radius: 0;
                                font-size: 14px;'>
                        <div style='font-weight: bold; font-size: 12px; margin-bottom: 4px;'>{sender_name}</div>
                        {text}
                    </div>
                    """

            html += "</div>"
            return html



        def load_user_chats():
            user_id = session.get("local_id")
            id_token = session.get("id_token")
            if not user_id:
                return gr.update(choices=[], value=None)

            # Load chat data
            chats_url = f"{DATABASE_URL}/chats.json?auth={id_token}"
            chats = requests.get(chats_url).json() or {}

            # Load user and resource data
            users_data = requests.get(f"{DATABASE_URL}/users.json?auth={id_token}").json() or {}
            resources_data = requests.get(f"{DATABASE_URL}/resources.json?auth={id_token}").json() or {}

            options = []
            chat_id_map.clear()

            for chat_id in chats:
                if user_id not in chat_id:
                    continue
                user1, user2 = chat_id.split("_")
                other_id = user2 if user1 == user_id else user1
                other_name = users_data.get(other_id, {}).get("username", "Unknown")

                # Try to infer resource name from their resources
                user_resources = resources_data.get(other_id, {})
                if user_resources:
                    first_resource = list(user_resources.values())[0]
                    resource_name = first_resource.get("name", "a resource")
                else:
                    resource_name = "a resource"

                label = f"Chat with {other_name} for {resource_name}"
                chat_id_map[label] = chat_id  # Map the label to the actual chat ID
                options.append(label)

            return gr.update(choices=options, value=None)

        def load_selected_chat(label):
            if not label or label not in chat_id_map:
                return gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
            
            chat_id = chat_id_map[label]
            session["current_chat_id"] = chat_id

            user_id = session["local_id"]  # ✅ Fetch the user_id from session

            # Display full conversation when "Load Chat" is clicked
            chat_html = get_messages_chat(chat_id, user_id)  # ✅ Now passing required arguments
            return gr.update(value=chat_html, visible=True), gr.update(visible=True), gr.update(visible=True)



        def send_message2(message):
            return send_message(message)

        # Hook up buttons and events
        refresh_chats_btn.click(load_user_chats, outputs=my_chats_dropdown)
        load_chat_btn.click(load_selected_chat, inputs=my_chats_dropdown, outputs=[chat_display2, message_box2, send_btn2])
        send_btn2.click(send_message2, inputs=message_box2, outputs=chat_display2)
    


    with gr.Tab("Save Changes"):
        description_text = gr.Markdown("Click 'Load My Resources' to edit their availability below:")
        load_button = gr.Button("Load My Resources")

        # Pre-create 10 checkboxes
        toggle_components = [gr.Checkbox(visible=False, label=f"Resource {i+1}") for i in range(10)]

        save_button = gr.Button("Save Changes")
        save_output = gr.Textbox(label="Save Output")

        toggle_data = []

        def load_toggles():
            uid = session.get("uid")
            id_token = session.get("id_token")
            username = session.get("username")
            if not uid or not id_token or not username:
                return [gr.update(visible=False)] * 10  # hide all

            try:
                url = f"{DATABASE_URL}/resources.json?auth={id_token}"
                res = requests.get(url)
                res.raise_for_status()
                all_resources = res.json()

                updates = []
                toggle_data.clear()

                idx = 0
                for user_id, resources in all_resources.items():
                    for resource_id, resource in resources.items():
                        if resource.get("owner") == username and idx < 10:
                            toggle_data.append((user_id, resource_id))
                            updates.append(gr.update(
                                label=resource.get("name", f"Resource {idx+1}"),
                                value=(resource.get("availability") == "available"),
                                visible=True
                            ))
                            idx += 1

                # Hide unused toggles
                while len(updates) < 10:
                    updates.append(gr.update(visible=False))

                return updates
            except Exception as e:
                print("Load error:", e)
                return [gr.update(visible=False)] * 10


        def save_toggles(*args):
            id_token = session.get("id_token")
            username = session.get("username")
            if not id_token or not username:
                return "Not logged in."

            try:
                for idx, state in enumerate(args):
                    if idx < len(toggle_data):
                        user_id, resource_id = toggle_data[idx]
                        availability = "available" if state else "unavailable"
                        url = f"{DATABASE_URL}/resources/{user_id}/{resource_id}.json?auth={id_token}"
                        requests.patch(url, json={"availability": availability})
                return "Updated successfully."
            except Exception as e:
                return f"Error: {e}"

        # Wire the buttons
        load_button.click(
            fn=load_toggles,
            outputs=toggle_components
        )

        save_button.click(
            fn=save_toggles,
            inputs=toggle_components,
            outputs=save_output
        )




demo.launch()
