#!/usr/bin/env python3
"""
Script to create a default admin user for the Health Risk Prediction System
Run this script to create an initial admin user account
"""

import sys
import os
sys.path.append('backend')

from backend.app import create_user

def main():
    print("🏥 Health Risk Prediction System - Admin User Creation")
    print("=" * 50)
    
    # Default admin credentials
    username = "admin"
    password = "admin123"
    email = "admin@healthsystem.com"
    full_name = "System Administrator"
    
    print(f"Creating admin user with username: {username}")
    print(f"Password: {password}")
    print(f"Email: {email}")
    print(f"Full Name: {full_name}")
    print()
    
    # Create the user
    success, message = create_user(username, password, email, full_name)
    
    if success:
        print("✅ Admin user created successfully!")
        print(f"Message: {message}")
        print()
        print("You can now login with:")
        print(f"Username: {username}")
        print(f"Password: {password}")
        print()
        print("⚠️  IMPORTANT: Please change the default password after first login!")
    else:
        print("❌ Failed to create admin user!")
        print(f"Error: {message}")
        
        if "already exists" in message.lower():
            print()
            print("The admin user already exists. You can login with:")
            print(f"Username: {username}")
            print(f"Password: {password}")

if __name__ == "__main__":
    main()
