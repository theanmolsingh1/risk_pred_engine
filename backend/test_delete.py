import requests
import json

def test_delete_functionality():
    """Test the delete patient functionality"""
    base_url = "http://localhost:5000"
    
    try:
        # First, get all patients to see what we have
        print("1. Getting all patients...")
        response = requests.get(f"{base_url}/api/patients")
        if response.status_code == 200:
            patients = response.json()
            print(f"✅ Found {len(patients)} patients")
            if patients:
                print("Sample patient:", json.dumps(patients[0], indent=2))
                
                # Test delete on first patient (if any exist)
                patient_id = patients[0]['patient_id']
                patient_name = patients[0]['name']
                
                print(f"\n2. Testing delete for patient: {patient_name} (ID: {patient_id})")
                delete_response = requests.delete(f"{base_url}/api/patients/{patient_id}")
                
                if delete_response.status_code == 200:
                    result = delete_response.json()
                    print(f"✅ Delete successful: {result['message']}")
                    
                    # Verify deletion by getting patients again
                    print("\n3. Verifying deletion...")
                    verify_response = requests.get(f"{base_url}/api/patients")
                    if verify_response.status_code == 200:
                        remaining_patients = verify_response.json()
                        print(f"✅ Remaining patients: {len(remaining_patients)}")
                        if len(remaining_patients) < len(patients):
                            print("✅ Patient successfully deleted!")
                        else:
                            print("❌ Patient still exists after deletion")
                    else:
                        print(f"❌ Error verifying deletion: {verify_response.text}")
                else:
                    result = delete_response.json()
                    print(f"❌ Delete failed: {result.get('error', 'Unknown error')}")
            else:
                print("ℹ️ No patients found to test deletion")
        else:
            print(f"❌ Error getting patients: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask app. Make sure it's running on localhost:5000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_delete_functionality()
