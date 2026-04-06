"""
Data Augmentation for Skin Condition Dataset
Creates diverse, high-quality augmented images for training
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import random

class DataAugmentation:
    def __init__(self, dataset_path, output_path, target_images_per_class=100):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.target_images_per_class = target_images_per_class
        self.img_size = (224, 224)
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Setup augmentation strategies
        self.setup_augmenters()
        
    def setup_augmenters(self):
        """Setup multiple augmentation strategies"""
        
        # Primary augmenter - diverse transformations
        self.primary_augmenter = ImageDataGenerator(
            rotation_range=60,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.4,
            zoom_range=[0.7, 1.3],
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.6, 1.4],
            channel_shift_range=0.15,
            fill_mode='reflect'
        )
        
        # Subtle augmenter - preserving key features
        self.subtle_augmenter = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=[0.9, 1.1],
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Extreme augmenter - for robustness
        self.extreme_augmenter = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.4,
            height_shift_range=0.4,
            shear_range=0.6,
            zoom_range=[0.5, 1.5],
            horizontal_flip=True,
            brightness_range=[0.5, 1.5],
            channel_shift_range=0.2,
            fill_mode='reflect'
        )
    
    def apply_pil_enhancements(self, image):
        """Apply PIL-based enhancements"""
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Random enhancements
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(random.uniform(0.8, 1.3))
        
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(random.uniform(0.8, 1.3))
        
        if random.random() > 0.5:
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(random.uniform(0.8, 1.3))
        
        if random.random() > 0.7:
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
        
        return np.array(pil_image) / 255.0
    
    def apply_cv2_enhancements(self, image):
        """Apply OpenCV-based enhancements"""
        img = (image * 255).astype(np.uint8)
        
        # CLAHE for contrast enhancement
        if random.random() > 0.5:
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img_lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img_lab[:,:,0])
            img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # Histogram equalization
        if random.random() > 0.5:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        # Add noise for robustness
        if random.random() > 0.7:
            noise = np.random.normal(0, 10, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img / 255.0
    
    def create_placeholder_images(self, output_path, class_name):
        """Create placeholder images when no originals exist"""
        print(f"   Creating placeholder images for {class_name}")
        
        # Different colors for each class
        colors = {
            'clear skin': [(255, 220, 177), (255, 200, 150)],
            'dark spots': [(150, 100, 50), (100, 60, 30)],
            'puffy eyes': [(200, 180, 160), (180, 160, 140)],
            'wrinkles': [(180, 150, 120), (160, 130, 100)]
        }
        
        class_colors = colors.get(class_name, [(128, 128, 128), (100, 100, 100)])
        
        for i in range(self.target_images_per_class):
            # Create base image
            img = np.random.randint(50, 100, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            
            # Add class-specific color
            base_color = random.choice(class_colors)
            img = img * 0.3 + np.array(base_color) * 0.7
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Add texture/pattern
            if 'wrinkles' in class_name:
                # Add line patterns
                for _ in range(random.randint(5, 15)):
                    y = random.randint(0, self.img_size[0])
                    x1, x2 = random.randint(0, self.img_size[1]), random.randint(0, self.img_size[1])
                    cv2.line(img, (x1, y), (x2, y), (50, 50, 50), 1)
            
            elif 'dark spots' in class_name:
                # Add spots
                for _ in range(random.randint(10, 30)):
                    x, y = random.randint(0, self.img_size[1]), random.randint(0, self.img_size[0])
                    radius = random.randint(2, 8)
                    cv2.circle(img, (x, y), radius, (80, 60, 40), -1)
            
            elif 'puffy eyes' in class_name:
                # Add circular patterns
                for _ in range(random.randint(2, 4)):
                    x, y = random.randint(50, self.img_size[1]-50), random.randint(50, self.img_size[0]-50)
                    radius = random.randint(20, 40)
                    cv2.circle(img, (x, y), radius, (150, 140, 130), -1)
            
            # Add noise and blur
            noise = np.random.normal(0, 10, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Save image
            output_file = os.path.join(output_path, f"placeholder_{i:04d}.jpg")
            pil_img = Image.fromarray(img)
            pil_img.save(output_file, 'JPEG', quality=85)
    
    def augment_class(self, class_name):
        """Augment images for a specific class"""
        class_input_path = os.path.join(self.dataset_path, class_name)
        class_output_path = os.path.join(self.output_path, class_name)
        
        if not os.path.isdir(class_input_path):
            print(f"⚠️  Class directory not found: {class_name}")
            return False
        
        # Create output directory
        os.makedirs(class_output_path, exist_ok=True)
        
        # Get original images
        original_images = []
        image_files = [f for f in os.listdir(class_input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        print(f"\n📁 Processing class: {class_name}")
        print(f"   Original images: {len(image_files)}")
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in {class_name}")
            # Create placeholder images
            self.create_placeholder_images(class_output_path, class_name)
            return True
        
        # Load original images
        for img_file in image_files:
            img_path = os.path.join(class_input_path, img_file)
            try:
                img = load_img(img_path, target_size=self.img_size)
                img_array = img_to_array(img) / 255.0
                original_images.append(img_array)
                
                # Copy original to output
                output_path = os.path.join(class_output_path, f"original_{img_file}")
                save_img(output_path, img_array)
            except Exception as e:
                print(f"⚠️  Error loading {img_file}: {e}")
        
        # Calculate needed augmented images
        current_count = len(original_images)
        needed_count = max(0, self.target_images_per_class - current_count)
        
        print(f"   Target images: {self.target_images_per_class}")
        print(f"   Need to create: {needed_count} augmented images")
        
        if needed_count > 0:
            # Create augmented images using multiple strategies
            augmented_images = []
            
            # Strategy 1: Keras augmentation (40%)
            keras_count = int(needed_count * 0.4)
            for i in range(keras_count):
                base_img = random.choice(original_images)
                img = np.expand_dims(base_img, axis=0)
                aug_img = self.primary_augmenter.random_transform(img[0])
                augmented_images.append(aug_img)
            
            # Strategy 2: PIL enhancements (30%)
            pil_count = int(needed_count * 0.3)
            for i in range(pil_count):
                base_img = random.choice(original_images)
                aug_img = self.apply_pil_enhancements(base_img)
                augmented_images.append(aug_img)
            
            # Strategy 3: OpenCV enhancements (20%)
            cv2_count = int(needed_count * 0.2)
            for i in range(cv2_count):
                base_img = random.choice(original_images)
                aug_img = self.apply_cv2_enhancements(base_img)
                augmented_images.append(aug_img)
            
            # Strategy 4: Mixed transformations (10%)
            mixed_count = needed_count - len(augmented_images)
            for i in range(mixed_count):
                base_img = random.choice(original_images)
                img = base_img.copy()
                
                # Apply multiple enhancements
                if random.random() > 0.5:
                    img = self.apply_pil_enhancements(img)
                if random.random() > 0.5:
                    img = self.apply_cv2_enhancements(img)
                
                # Apply Keras augmentation
                img = np.expand_dims(img, axis=0)
                img = self.primary_augmenter.random_transform(img[0])
                augmented_images.append(img)
            
            # Save augmented images
            for i, aug_img in enumerate(augmented_images):
                output_path = os.path.join(class_output_path, f"aug_{i:04d}.jpg")
                save_img(output_path, aug_img)
        
        final_count = len([f for f in os.listdir(class_output_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   ✅ Final count: {final_count} images")
        
        return True
    
    def augment_dataset(self):
        """Augment entire dataset"""
        print("🚀 Starting Data Augmentation")
        print("="*50)
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path not found: {self.dataset_path}")
            return False
        
        # Get all classes
        classes = [d for d in os.listdir(self.dataset_path) 
                  if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        print(f"📁 Found {len(classes)} classes: {classes}")
        
        # Process each class
        processed_classes = []
        for class_name in classes:
            try:
                success = self.augment_class(class_name)
                if success:
                    processed_classes.append(class_name)
            except Exception as e:
                print(f"❌ Error processing {class_name}: {e}")
        
        # Final statistics
        print("\n" + "="*50)
        print("📊 AUGMENTATION RESULTS")
        print("="*50)
        
        total_images = 0
        for class_name in processed_classes:
            class_path = os.path.join(self.output_path, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_images += count
                print(f"   ✅ {class_name}: {count} images")
        
        print(f"\n🎯 Total images created: {total_images}")
        print(f"📁 Output directory: {self.output_path}")
        print("✅ Augmentation completed successfully!")
        
        return True

def main():
    """Main execution function"""
    print("🎨 Data Augmentation for Skin Conditions")
    print("="*40)
    
    # Configuration
    dataset_path = r"C:\Akshaya\internship_infosys\sample_folder"
    output_path = r"C:\Akshaya\internship_infosys\augmented_images"
    target_images_per_class = 100
    
    # Create augmenter
    augmenter = DataAugmentation(
        dataset_path=dataset_path,
        output_path=output_path,
        target_images_per_class=target_images_per_class
    )
    
    # Run augmentation
    success = augmenter.augment_dataset()
    
    if success:
        print("\n🚀 Ready for model training!")
        print("Run the EfficientNet classifier next.")

if __name__ == "__main__":
    main()