"""Modern mosaic generator with improved algorithms and caching."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal

import cv2
import numpy as np
from PIL import Image


class MosaicGenerator:
    """Generate mosaic images from text with modern Python practices."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_size: int = 1024,
        min_resolution: int = 64,
        max_resolution: int = 2048,
        default_tile_shape: Literal["square", "circle"] = "square",
    ):
        """Initialize the mosaic generator.
        
        Args:
            cache_dir: Directory to cache color dictionaries
            default_size: Default output image size
            min_resolution: Minimum mosaic resolution
            max_resolution: Maximum mosaic resolution
            default_tile_shape: Default shape for individual tiles ('square' or 'circle')
        """
        self.cache_dir = cache_dir or Path.home() / ".moji_mosaic" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_size = default_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.default_tile_shape = default_tile_shape
        self._color_cache: Dict[str, Tuple[int, int, int]] = {}
        
    def _load_color_cache(self, text_hash: str) -> Dict[str, Tuple[int, int, int]]:
        """Load color dictionary from cache."""
        cache_file = self.cache_dir / f"colors_{text_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {k: tuple(v) for k, v in data.items()}
            except (json.JSONDecodeError, ValueError):
                pass
        return {}
    
    def _save_color_cache(self, text_hash: str, colors: Dict[str, Tuple[int, int, int]]) -> None:
        """Save color dictionary to cache."""
        cache_file = self.cache_dir / f"colors_{text_hash}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({k: list(v) for k, v in colors.items()}, f, ensure_ascii=False)
        except IOError:
            pass  # Fail silently if caching fails
    
    def _calculate_mosaic_dimensions(self, text_length: int) -> Tuple[int, int]:
        """Calculate optimal mosaic dimensions based on text length."""
        # Make much smaller grids so tiles appear larger when scaled to 1024x1024
        # This dramatically reduces white space
        
        if text_length <= 4:
            side_length = 2  # 2x2 grid for very short text
        elif text_length <= 16:
            side_length = 4  # 4x4 grid 
        elif text_length <= 64:
            side_length = 8  # 8x8 grid
        elif text_length <= 256:
            side_length = 16  # 16x16 grid
        elif text_length <= 1024:
            side_length = 32  # 32x32 grid
        else:
            side_length = 64  # 64x64 grid for very long text
        
        return side_length, side_length
    
    def _draw_tile(self, image: np.ndarray, x: int, y: int, color: Tuple[int, int, int], 
                   tile_size: int, tile_shape: str = "square") -> None:
        """Draw a single tile (square or circle) on the image.
        
        Args:
            image: The image array to draw on
            x: X coordinate of tile center
            y: Y coordinate of tile center  
            color: RGB color tuple
            tile_size: Size of the tile
            tile_shape: Shape of the tile ('square' or 'circle')
        """
        height, width = image.shape[:2]
        half_size = tile_size // 2
        
        if tile_shape == "circle":
            # Draw circular tile with margin (smaller circle)
            # Reduce circle radius by 35% to create more margin
            circle_radius = int(half_size * 0.65)
            for dy in range(-half_size, half_size + 1):
                for dx in range(-half_size, half_size + 1):
                    px, py = x + dx, y + dy
                    if (0 <= px < width and 0 <= py < height and 
                        dx*dx + dy*dy <= circle_radius*circle_radius):
                        image[py, px] = color
        else:
            # Draw square tile (original behavior)
            x1, y1 = max(0, x - half_size), max(0, y - half_size)
            x2, y2 = min(width, x + half_size + 1), min(height, y + half_size + 1)
            image[y1:y2, x1:x2] = color
    
    def _generate_unique_color(self, existing_colors: set) -> Tuple[int, int, int]:
        """Generate a unique RGB color not in the existing set."""
        max_attempts = 1000
        for _ in range(max_attempts):
            color = tuple(int(x) for x in np.random.randint(0, 256, 3))
            if color not in existing_colors:
                return color
        
        # Fallback: generate systematic colors if random fails
        for r in range(0, 256, 32):
            for g in range(0, 256, 32):
                for b in range(0, 256, 32):
                    color = (r, g, b)
                    if color not in existing_colors:
                        return color
        
        # Ultimate fallback
        return (int(np.random.randint(0, 256)), int(np.random.randint(0, 256)), int(np.random.randint(0, 256)))
    
    def _assign_colors(self, text: str) -> Dict[str, Tuple[int, int, int]]:
        """Assign unique colors to each character in the text."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        color_dict = self._load_color_cache(text_hash)
        
        unique_chars = set(text)
        existing_colors = set(color_dict.values())
        
        # Assign colors to new characters
        for char in unique_chars:
            if char not in color_dict:
                color_dict[char] = self._generate_unique_color(existing_colors)
                existing_colors.add(color_dict[char])
        
        self._save_color_cache(text_hash, color_dict)
        return color_dict
    
    def _create_mosaic_array(
        self, 
        text: str, 
        color_dict: Dict[str, Tuple[int, int, int]], 
        grid_width: int, 
        grid_height: int,
        output_size: int,
        tile_shape: str = "square"
    ) -> np.ndarray:
        """Create the mosaic array with randomized character placement.
        
        Args:
            text: Input text
            color_dict: Character to color mapping
            grid_width: Grid width (number of tiles)
            grid_height: Grid height (number of tiles)
            output_size: Final output image size
            tile_shape: Shape of individual tiles ('square' or 'circle')
            
        Returns:
            Mosaic array as numpy ndarray
        """
        # Initialize white canvas
        mosaic = np.full((output_size, output_size, 3), 255, dtype=np.uint8)
        
        # Calculate tile size
        tile_width = output_size // grid_width
        tile_height = output_size // grid_height
        tile_size = min(tile_width, tile_height)
        
        # Generate all possible grid positions
        positions = [(i, j) for i in range(grid_height) for j in range(grid_width)]
        np.random.shuffle(positions)
        
        # Place characters as tiles
        for idx, char in enumerate(text):
            if idx >= len(positions):
                break  # More characters than positions
            
            grid_y, grid_x = positions[idx]
            
            # Convert grid position to pixel position (center of tile)
            pixel_x = grid_x * tile_width + tile_width // 2
            pixel_y = grid_y * tile_height + tile_height // 2
            
            # Draw the tile
            self._draw_tile(mosaic, pixel_x, pixel_y, color_dict[char], tile_size, tile_shape)
        
        return mosaic
    
    def calculate_tile_size(self, text: str, output_size: int) -> int:
        """Calculate the tile size that would be used for given text and output size.
        
        Args:
            text: Input text
            output_size: Output image size
            
        Returns:
            Calculated tile size in pixels
        """
        cleaned_text = text.replace('\n', '').replace('\r', '')
        if len(cleaned_text) > 1_048_576:
            cleaned_text = cleaned_text[:1_048_576]
            
        width, height = self._calculate_mosaic_dimensions(len(cleaned_text))
        
        # Calculate tile size (same logic as in _create_mosaic_array)
        tile_width = output_size // width
        tile_height = output_size // height
        tile_size = min(tile_width, tile_height)
        
        return tile_size
    
    def generate_mosaic(
        self, 
        text: str, 
        output_size: Optional[int] = None,
        preserve_aspect_ratio: bool = True,
        tile_shape: Optional[Literal["square", "circle"]] = None
    ) -> Tuple[Image.Image, int]:
        """Generate a mosaic image from input text.
        
        Args:
            text: Input text to convert to mosaic
            output_size: Final output image size (default: self.default_size)
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            tile_shape: Shape of individual tiles ('square' or 'circle', default: self.default_tile_shape)
            
        Returns:
            Tuple of (PIL Image of the generated mosaic, actual tile size used)
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Clean text (remove newlines, normalize)
        cleaned_text = text.replace('\n', '').replace('\r', '')
        if len(cleaned_text) > 1_048_576:  # Limit for performance
            cleaned_text = cleaned_text[:1_048_576]
        
        # Calculate mosaic dimensions
        width, height = self._calculate_mosaic_dimensions(len(cleaned_text))
        
        # Assign colors to characters
        color_dict = self._assign_colors(cleaned_text)
        
        # Use specified tile shape or default
        tile_shape = tile_shape or self.default_tile_shape
        
        # Get output size
        output_size = output_size or self.default_size
        
        # Create mosaic array with proper tile rendering
        mosaic_array = self._create_mosaic_array(cleaned_text, color_dict, width, height, output_size, tile_shape)
        
        # Convert to PIL Image (OpenCV uses BGR, PIL uses RGB)
        rgb_array = cv2.cvtColor(mosaic_array, cv2.COLOR_BGR2RGB)
        
        # Calculate the actual tile size used
        tile_size = self.calculate_tile_size(text, output_size)
        
        return Image.fromarray(rgb_array), tile_size
    
    def get_color_statistics(self, text: str) -> Dict:
        """Get statistics about the color distribution in the text."""
        color_dict = self._assign_colors(text)
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        return {
            'unique_characters': len(color_dict),
            'total_characters': len(text),
            'character_frequency': char_counts,
            'color_palette': color_dict,
            'most_frequent_char': max(char_counts.items(), key=lambda x: x[1]) if char_counts else None,
        }
