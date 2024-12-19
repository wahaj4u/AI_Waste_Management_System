import streamlit as st
import requests
from PIL import Image

# Title of the Streamlit App
st.title("Waste Classification App")
st.write("Capture an image using your camera to classify the waste and get disposal instructions.")

# Disposal recommendation dictionary
disposal_recommendations = {
    "aerosol_cans": {
        "recommendation": "Make sure the can is empty before disposal. Check with your local recycling program for acceptance. If not recyclable, dispose of as hazardous waste.",
        "type": "Hazardous Waste",
        "recyclable": "No",
        "compostable": "No"
    },
    "aluminum_food_cans": {
        "recommendation": "Rinse the can thoroughly to remove any food residue. Place it in your recycling bin. Crushing the can saves space but is optional.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "aluminum_soda_cans": {
        "recommendation": "Rinse to remove sticky residue. Place the can in your recycling bin. Avoid crushing if your recycling program requires intact cans.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "cardboard_boxes": {
        "recommendation": "Flatten the box to save space before recycling. Remove any non-cardboard elements like tape or labels. Place in the recycling bin for paper/cardboard.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "cardboard_packaging": {
        "recommendation": "Ensure all packaging is flattened for easy recycling. Remove non-cardboard parts such as plastic film or foam. Recycle with other cardboard materials.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "clothing": {
        "recommendation": "If still wearable, consider donating to local charities or thrift stores. For damaged clothing, recycle as fabric or take to textile recycling bins. Avoid placing in general waste.",
        "type": "Recyclable/Textile Recycling",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "coffee_grounds": {
        "recommendation": "Coffee grounds are rich in nutrients and can be composted. Add them to your compost bin or garden soil. If composting is not an option, dispose of them in organic waste bins.",
        "type": "Compostable",
        "recyclable": "No",
        "compostable": "Yes"
    },
    "disposable_plastic_cutlery": {
        "recommendation": "Most disposable cutlery is not recyclable. Place it in the general waste bin. Consider switching to reusable or compostable alternatives in the future.",
        "type": "General Waste",
        "recyclable": "No",
        "compostable": "No"
    },
    "eggshells": {
        "recommendation": "Eggshells can be composted and are great for enriching soil. Add them to your compost bin after rinsing. Alternatively, place in organic waste bins.",
        "type": "Compostable",
        "recyclable": "No",
        "compostable": "Yes"
    },
    "food_waste": {
        "recommendation": "Separate food waste from packaging before disposal. Compost if possible to reduce landfill impact. Use organic waste bins where available.",
        "type": "Compostable",
        "recyclable": "No",
        "compostable": "Yes"
    },
    "glass_beverage_bottles": {
        "recommendation": "Rinse thoroughly to remove any liquid. Place in the glass recycling bin. Remove caps or lids if not made of glass.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "glass_cosmetic_containers": {
        "recommendation": "Clean the container to ensure it's residue-free. Recycle if your local program accepts glass containers. Broken glass should be wrapped in paper or cardboard and placed in general waste.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "glass_food_jars": {
        "recommendation": "Rinse the jar to remove food residue. Recycle in glass bins. Lids made of metal can often be recycled separately.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "magazines": {
        "recommendation": "Remove plastic covers or non-paper elements before recycling. Place in your paper recycling bin. Avoid recycling if excessively wet or damaged.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "newspaper": {
        "recommendation": "Keep newspapers dry and free of contaminants like food stains. Recycle them in designated paper bins. Bundle them for easier handling if required.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "office_paper": {
        "recommendation": "Shred confidential documents if necessary before recycling. Avoid including paper with heavy lamination or plastic content. Recycle in paper bins.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "paper_cups": {
        "recommendation": "Check for a recycling symbol to confirm if recyclable. Most paper cups with plastic lining are not recyclable and go into general waste. Consider switching to reusable cups.",
        "type": "General Waste",
        "recyclable": "No",
        "compostable": "No"
    },
    "plastic_cup_lids": {
        "recommendation": "If marked recyclable, clean and place them in the appropriate bin. Otherwise, dispose of in general waste. Avoid using single-use lids when possible.",
        "type": "Recyclable/General Waste",
        "recyclable": "Yes/No",
        "compostable": "No"
    },
    "plastic_detergent_bottles": {
        "recommendation": "Rinse out any remaining detergent to avoid contamination. Check the recycling symbol and place in plastics recycling. Keep the lid on if acceptable.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "plastic_food_containers": {
        "recommendation": "Ensure the container is clean and free of food residue. Recycle if marked as recyclable. Otherwise, dispose of in general waste.",
        "type": "Recyclable/General Waste",
        "recyclable": "Yes/No",
        "compostable": "No"
    },
    "plastic_shopping_bags": {
        "recommendation": "Reuse them for storage or garbage liners. If recycling facilities for plastic bags are available, drop them off. Avoid throwing in general recycling bins.",
        "type": "Recyclable/General Waste",
        "recyclable": "Yes/No",
        "compostable": "No"
    },
    "plastic_soda_bottles": {
        "recommendation": "Empty and rinse the bottle before recycling. Leave the cap on if your recycling program accepts it. Crush the bottle to save space if desired.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "plastic_straws": {
        "recommendation": "Plastic straws are not recyclable in most programs. Dispose of them in general waste. Consider using reusable or biodegradable straws.",
        "type": "General Waste",
        "recyclable": "No",
        "compostable": "No"
    },
    "plastic_trash_bags": {
        "recommendation": "Trash bags themselves are not recyclable. Dispose of them in general waste along with their contents. Look for biodegradable options when purchasing new ones.",
        "type": "General Waste",
        "recyclable": "No",
        "compostable": "No"
    },
    "plastic_water_bottles": {
        "recommendation": "Rinse the bottle to ensure cleanliness. Recycle the bottle along with the cap if accepted. Try to use reusable bottles to reduce plastic waste.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "shoes": {
        "recommendation": "Donate shoes that are still wearable to charities or thrift stores. For damaged or unusable shoes, check for textile recycling bins. Avoid discarding in general waste.",
        "type": "Recyclable/Textile Recycling",
        "recyclable": "Yes",
        "compostable": "No"
    }
}


# Camera input section
captured_image = st.camera_input("Capture an image using your camera")

if captured_image is not None:
    try:
        # Display the captured image
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        # Convert the captured image to bytes for the POST request
        st.write("Classifying the waste...")
        files = {"image": captured_image.getvalue()}
        
        # Replace with the active ngrok or backend URL
        backend_url = "https://1e6f-34-145-73-108.ngrok-free.app/process"

        # Send the image to the Flask backend
        response = requests.post(backend_url, files=files)

        if response.status_code == 200:
            result = response.json()
            category = result.get('category', 'unknown').lower()
            
            # Fetch disposal information
            waste_info = disposal_methods.get(category, {
                "recommendation": "No specific instructions available.",
                "type": "Unknown",
                "recyclable": "Unknown",
                "compostable": "Unknown"
            })
            
            # Display the results
            st.write("### Classification Result:")
            st.write(f"**Category**: {result.get('category', 'N/A')}")
            st.write(f"**Type**: {waste_info['type']}")
            st.write(f"**Recyclable**: {waste_info['recyclable']}")
            st.write(f"**Compostable**: {waste_info['compostable']}")
            st.write(f"**Disposal Recommendation**: {waste_info['recommendation']}")
        else:
            st.error(f"Error: Unable to classify the image. Status code: {response.status_code}")
            st.error(f"Response: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend. Please ensure the backend is running and the URL is correct.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
