# Initialize SAM model
def load_sam_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'sam_vit_b.pth'

    # Google Drive file ID from the link
    google_drive_file_id = '1_KcHtbr2x7wxCHRLKn1r5jSUY_wK3POe'  # Replace with your actual file ID
    google_drive_url = f'https://drive.google.com/uc?id={google_drive_file_id}'

    # Check if the model is already downloaded
    if not os.path.exists(model_path):
        print(f"Downloading model from Google Drive: {google_drive_url}")
        
        # Download the model from Google Drive using gdown
        gdown.download(google_drive_url, model_path, quiet=False)

    if os.path.exists(model_path):
        try:
            print(f"Loading model from: {model_path}")
            
            # Load the model state_dict
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Now load the model with the state_dict
            sam = sam_model_registry['vit_b'](checkpoint=state_dict)
            
            sam.to(device)
            return SamAutomaticMaskGenerator(sam)
        except Exception as e:
            print(f"Error loading the model: {e}")
            st.error(f"Failed to load the model: {e}")
    else:
        st.error("Model file could not be downloaded!")
