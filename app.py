from flask import Flask, request, jsonify

# Define categories and corresponding disposal recommendations
disposal_methods = {
    "aerosol_cans": {"recommendation": "Recycle at a designated facility.", "type": "recyclable"},
    "aluminum_soda_cans": {"recommendation": "Recycle at a designated facility.", "type": "recyclable"},
    "cardboard_boxes": {"recommendation": "Recycle at a designated facility.", "type": "recyclable"},
    "cardboard_packaging": {"recommendation": "Recycle at a designated facility.", "type": "recyclable"},
    "clothing": {"recommendation": "Donate or recycle at textile recycling center.", "type": "recyclable"},
    "coffee_grounds": {"recommendation": "Compost or dispose of in organic waste.", "type": "organic"},
    "disposable_plastic_cutlery": {"recommendation": "Dispose of in trash as non-recyclable.", "type": "non-recyclable"},
    "eggshells": {"recommendation": "Compost or dispose of in organic waste.", "type": "organic"},
    "food_waste": {"recommendation": "Compost or dispose of in organic waste.", "type": "organic"},
    "glass_bottles": {"recommendation": "Recycle at a designated facility.", "type": "recyclable"},
    # Add all categories here as needed...
}

# Flask app setup
app = Flask(__name__)

# Endpoint to get disposal information based on keyword
@app.route('/process', methods=['GET'])
def get_disposal_info():
    # Extract the 'keyword' from the query parameters
    keyword = request.args.get('keyword', '').lower()  # Convert to lowercase to make the search case-insensitive
    
    # Lookup the keyword in the disposal methods dictionary
    disposal_info = disposal_methods.get(keyword)
    
    if not disposal_info:
        return jsonify({'error': 'No disposal information available for the provided keyword'}), 404
    
    return jsonify({
        "category": keyword,
        "disposal_recommendation": disposal_info["recommendation"],
        "type": disposal_info["type"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
