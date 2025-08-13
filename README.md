# Fitness AI Exercise Classification System
*A Deep Learning Solution for Automated Workout Recognition*

## 📌 Overview
This project presents an AI-powered system that automatically recognizes **22 different exercise types** from workout videos.  
It combines **pose estimation**, **biomechanical feature extraction**, and **sequence modeling** to deliver robust, real-world exercise classification — even in challenging conditions such as varying lighting, camera angles, and backgrounds.

The model is designed for:
- **Personal fitness tracking**
- **Virtual coaching & online training platforms**
- **Rehabilitation and sports performance analysis**

---

## 📊 Dataset
- **Source:** Public workout videos from YouTube  
- **Categories:** 14 upper body, 6 lower body, and 2 core exercises  
- **Diversity:** Multiple environments, camera angles, and participant demographics  
- **Challenges:** Class imbalance, partial body coverage, motion blur, and background distractions

---

## 📈 Key Results
- **Test Accuracy:** 79.1%  
- **Balanced Accuracy:** 72.8%  
- **Cohen’s Kappa:** 0.792 (substantial agreement)  
- **High performers:** Push-Up (92%), Tricep Pushdown (94%), Lateral Raise (88%)  
- **Challenging cases:** Romanian Deadlift, Hammer Curl, Pull-Up  

The system demonstrates strong performance for most exercises and provides realistic generalization to unseen videos.

---

## 🚀 How It Works (High-Level)
1. **Pose Estimation** – Detect human body landmarks from video frames using MediaPipe  
2. **Feature Extraction** – Calculate biomechanical features such as joint angles, distances, and movement velocities  
3. **Sequence Analysis** – Model temporal patterns in exercise execution  
4. **Prediction & Confidence Scoring** – Output the most likely exercise type with a confidence score  

---

## 💡 Key Contributions
- Handles **real-world video variability** without requiring controlled environments  
- Addresses **class imbalance** using focal loss and class weighting  
- Uses **confidence-based filtering** to remove low-quality pose detections  
- Employs **ensemble predictions** for robust final classification  

---

## 🔮 Future Directions
- Add **vision-language models** for context-aware recognition  
- Combine **pose data with visual cues** for better equipment-based classification  
- Deploy on **mobile and edge devices** for real-time exercise tracking  
- Extend to **form assessment and injury prevention feedback**

---

## 📬 Contact
**Author:** Hay Lahav  
📧 Email: haylahav1@gmail.com
