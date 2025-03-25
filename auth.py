# auth.py
import hmac
import hashlib
import base64
import streamlit as st
import logging

logger = logging.getLogger(__name__)

def get_secret_hash(username, client_id, client_secret):
    message = username + client_id
    dig = hmac.new(
        client_secret.encode('utf-8'),
        msg=message.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(dig).decode()

def authenticate_user(cognito_client, username, password):
    secret_hash = get_secret_hash(username, st.secrets["COGNITO_CLIENT_ID"], st.secrets["COGNITO_CLIENT_SECRET"])
    try:
        response = cognito_client.initiate_auth(
            ClientId=st.secrets["COGNITO_CLIENT_ID"],
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password,
                'SECRET_HASH': secret_hash
            }
        )
        return response['AuthenticationResult']
    except cognito_client.exceptions.NotAuthorizedException:
        st.error("Incorrect username or password.")
    except cognito_client.exceptions.UserNotFoundException:
        st.error("User does not exist.")
    except Exception as e:
        st.error(f"An error occurred during authentication: {e}")
        logger.error(f"Authentication Error: {e}")
    return None

def handle_login(cognito_client, get_username_by_email_func, email, password):
    if email and password:
        username = get_username_by_email_func(email)
        if username:
            auth_result = authenticate_user(cognito_client, username, password)
            if auth_result:
                st.session_state['access_token'] = auth_result['AccessToken']
                st.session_state['username'] = username
                st.session_state['user_email'] = email
                return True
            else:
                return False
        else:
            st.error("Username retrieval failed.")
            return False
    else:
        st.error("Please enter both email and password.")
        return False