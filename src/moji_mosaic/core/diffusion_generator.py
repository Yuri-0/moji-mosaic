"""Diffusion model-based artistic image generation using state-of-the-art models."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForImage2Image


class DiffusionGenerator:
    """Generate artistic images using state-of-the-art diffusion models."""
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        use_vllm: bool = True,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        enable_cpu_offload: bool = False,
        enable_attention_slicing: bool = False,
    ):
        """Initialize the diffusion generator.
        
        Args:
            model_name: HuggingFace model name for diffusion model
            use_vllm: Whether to use vLLM for text processing
            device: Device to run the model on (auto-detected if None)
            cache_dir: Directory to cache models
            enable_cpu_offload: Enable CPU offloading for memory efficiency
            enable_attention_slicing: Enable attention slicing for memory efficiency
        """
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.cache_dir = cache_dir or Path.home() / ".moji_mosaic" / "diffusion_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_attention_slicing = enable_attention_slicing
        
        # Initialize models
        self._pipeline = None
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize the diffusion pipeline and optional vLLM model."""
        try:
            logging.info(f"Loading diffusion model: {self.model_name}")

            # Load the img2img pipeline for mosaic transformation
            self._pipeline = AutoPipelineForImage2Image.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                cache_dir=str(self.cache_dir),
            )

            # Move to device
            if self.device != "cpu":
                self._pipeline = self._pipeline.to(self.device)

            # Memory optimizations
            if self.enable_cpu_offload and hasattr(self._pipeline, 'enable_model_cpu_offload'):
                self._pipeline.enable_model_cpu_offload()

            if self.enable_attention_slicing and hasattr(self._pipeline, 'enable_attention_slicing'):
                self._pipeline.enable_attention_slicing()

            logging.info("Diffusion pipeline loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize diffusion models: {e}")
            self._pipeline = None
    
    def _create_keywords_prompt(self, keywords: List[str]) -> str:
        """Create a prompt string from keywords."""
        if not keywords:
            return "abstract composition"
        
        return ", ".join(f"'{keyword}'" for keyword in keywords)
    
    
    def _prepare_mosaic_image(self, mosaic_image: Image.Image, target_size: int = 512) -> Image.Image:
        """Prepare the mosaic image for diffusion processing."""
        # Resize to target size while maintaining aspect ratio
        mosaic_image = mosaic_image.convert("RGB")
        
        # Calculate new dimensions
        width, height = mosaic_image.size
        if width != height:
            # Make it square by padding
            max_dim = max(width, height)
            new_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            paste_x = (max_dim - width) // 2
            paste_y = (max_dim - height) // 2
            new_image.paste(mosaic_image, (paste_x, paste_y))
            mosaic_image = new_image
        
        # Resize to target size
        mosaic_image = mosaic_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        return mosaic_image
    
    def _is_white_background(self, color: np.ndarray, threshold: int = 240) -> bool:
        """Check if a color is considered white background.
        
        Args:
            color: RGB color array
            threshold: Minimum value for each RGB channel to be considered white
            
        Returns:
            True if the color is considered white background
        """
        return np.all(color >= threshold)
    
    def _enhance_colors_from_mosaic(
        self, 
        generated_image: Image.Image, 
        mosaic_image: Image.Image, 
        tile_size: int = 16,
        enhancement_strength: float = 0.3
    ) -> Image.Image:
        """Enhance colors in the generated image based on mosaic tile colors.
        
        Args:
            generated_image: The diffusion-generated image
            mosaic_image: The original mosaic image
            tile_size: Size of each mosaic tile for color mapping
            enhancement_strength: Strength of color enhancement (0.0-1.0)
            
        Returns:
            Enhanced image with mosaic colors emphasized
        """
        # Convert images to numpy arrays
        gen_array = np.array(generated_image)
        mosaic_array = np.array(mosaic_image)
        
        # Ensure both images have the same size
        if gen_array.shape[:2] != mosaic_array.shape[:2]:
            mosaic_pil = Image.fromarray(mosaic_array)
            mosaic_pil = mosaic_pil.resize(generated_image.size, Image.Resampling.LANCZOS)
            mosaic_array = np.array(mosaic_pil)
        
        height, width = gen_array.shape[:2]
        enhanced_array = gen_array.copy().astype(np.float32)
        
        # Calculate grid dimensions
        grid_height = height // tile_size
        grid_width = width // tile_size
        
        # Process each grid tile
        for row in range(grid_height):
            for col in range(grid_width):
                # Calculate tile boundaries
                y_start = row * tile_size
                y_end = min((row + 1) * tile_size, height)
                x_start = col * tile_size
                x_end = min((col + 1) * tile_size, width)
                
                # Get the average color of the mosaic tile
                mosaic_tile = mosaic_array[y_start:y_end, x_start:x_end]
                mosaic_color = np.mean(mosaic_tile, axis=(0, 1))
                
                # Skip if the mosaic color is white background
                if self._is_white_background(mosaic_color):
                    continue
                
                # Get the corresponding tile in the generated image
                gen_tile = enhanced_array[y_start:y_end, x_start:x_end]
                
                # Create a mask to avoid enhancing white areas in the generated image
                white_mask = np.all(gen_tile >= 240, axis=2)
                
                # Apply color enhancement only to non-white areas
                for c in range(3):  # RGB channels
                    # Blend the mosaic color with the generated image
                    enhanced_channel = (
                        gen_tile[:, :, c] * (1 - enhancement_strength) + 
                        mosaic_color[c] * enhancement_strength
                    )
                    
                    # Apply only to non-white areas
                    gen_tile[~white_mask, c] = enhanced_channel[~white_mask]
                
                # Update the enhanced array
                enhanced_array[y_start:y_end, x_start:x_end] = gen_tile
        
        # Convert back to uint8 and return as PIL Image
        enhanced_array = np.clip(enhanced_array, 0, 255).astype(np.uint8)
        return Image.fromarray(enhanced_array)
    
    def generate_artistic_image(
        self,
        mosaic_image: Image.Image,
        keywords: List[str],
        strength: float = 0.7,
        guidance_scale: float = 100,
        num_inference_steps: int = 200,
        seed: Optional[int] = None,
        tile_size: Optional[int] = None,
    ) -> Optional[Image.Image]:
        """Generate an artistic image from mosaic using only extracted keywords.
        
        Args:
            mosaic_image: Input mosaic image
            keywords: List of keywords to guide generation (no style additions)
            strength: How much to transform the input image (0.0-1.0)
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            tile_size: Size of mosaic tiles for color enhancement (auto-calculated if None)
            
        Returns:
            Generated artistic image or None if generation fails
        """
        if not self._pipeline:
            logging.error("Diffusion pipeline not available")
            return None
        
        try:
            # Prepare the mosaic image
            prepared_image = self._prepare_mosaic_image(mosaic_image)

            # Create prompt using only keywords - no style additions
            prompt = self._create_keywords_prompt(keywords)

            # Create transformation prompt
            prompt = ("transform every single tile to the cute objects with white background, which are concept "
                      f"about {prompt}. each pieces are must keep original color, and never including any text.")

            # Negative prompt to avoid unwanted elements
            negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, text, watermark"

            # Set random seed if provided
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)

            logging.info(f"Generating image with keywords: {prompt}")

            # Generate the image
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=prepared_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

            if result and result.images:
                generated_img = result.images[0]
                img = np.array(generated_img)

                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

                # 흰색에 가까운 영역 정의 (채도가 낮고 명도가 높은 영역)
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 30, 255])

                # 배경 마스크 생성
                mask = cv2.inRange(hsv, lower_white, upper_white)

                # 모폴로지 연산으로 마스크 정리
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # 배경을 순백으로 교체
                result = img.copy()
                result[mask > 0] = [255, 255, 255]
                
                # Apply color enhancement from mosaic
                final_img = Image.fromarray(result)
                
                # Calculate tile size if not provided
                if tile_size is None:
                    # Estimate tile size based on image dimensions
                    # This is a fallback - ideally tile_size should be passed from mosaic generation
                    image_size = prepared_image.size[0]  # Assuming square image
                    estimated_grid_size = int(np.sqrt(len([k for k in keywords if k])))
                    estimated_grid_size = max(8, min(64, estimated_grid_size))  # Reasonable bounds
                    tile_size = image_size // estimated_grid_size
                
                enhanced_img = self._enhance_colors_from_mosaic(
                    final_img, 
                    prepared_image,  # Use the prepared mosaic image
                    tile_size=tile_size,
                    enhancement_strength=0.8  # Adjust enhancement strength as needed
                )

                return enhanced_img
            
        except Exception as e:
            logging.error(f"Image generation failed: {e}")
            return None
        
        return None
    
    def generate_variations(
        self,
        mosaic_image: Image.Image,
        keywords: List[str],
        num_variations: int = 3,
        **generation_kwargs
    ) -> List[Image.Image]:
        """Generate multiple variations of the mosaic using only keywords."""
        variations = []
        
        for i in range(num_variations):
            # Use different seeds for variations
            seed = generation_kwargs.get('seed', None)
            if seed is not None:
                generation_kwargs['seed'] = seed + i
            
            # Vary the strength slightly for each variation
            base_strength = generation_kwargs.get('strength', 0.7)
            generation_kwargs['strength'] = base_strength + (i * 0.1 - 0.1)
            generation_kwargs['strength'] = max(0.3, min(0.9, generation_kwargs['strength']))
            
            image = self.generate_artistic_image(
                mosaic_image, keywords, **generation_kwargs
            )
            
            if image:
                variations.append(image)
        
        return variations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        return {
            'diffusion_model': self.model_name,
            'device': self.device,
            'pipeline_loaded': self._pipeline is not None,
            'memory_optimizations': {
                'cpu_offload': self.enable_cpu_offload,
                'attention_slicing': self.enable_attention_slicing,
            }
        }
