import pickle
import numpy as np
import os

similarity_path = "similarity.pkl"
output_path = "similarity_optimized.pkl"

if os.path.exists(similarity_path):
    print(f"🔄 Loading {similarity_path}...")
    with open(similarity_path, "rb") as f:
        similarity = pickle.load(f)
    
    print(f"📊 Original type: {similarity.dtype}, size: {os.path.getsize(similarity_path) / (1024*1024):.2f} MB")
    
    # Cast to float32 to reduce size by 50% from float64
    similarity_optimized = similarity.astype(np.float32)
    
    print(f"🚀 Optimized type: {similarity_optimized.dtype}")
    
    with open(output_path, "wb") as f:
        pickle.dump(similarity_optimized, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Saved to {output_path}, size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

    # Replace original if optimized is significantly smaller
    if os.path.getsize(output_path) < os.path.getsize(similarity_path):
        os.replace(output_path, similarity_path)
        print("♻️ Replaced original with optimized version")
else:
    print(f"❌ {similarity_path} not found")
