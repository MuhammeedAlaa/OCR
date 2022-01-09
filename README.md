# OCR

Small ocr model to get the text from photos

## PipleLine
1. First Technique 
    * Convert image to greyscale
    * Remove noise 
    * Apply Otsu thresholding
    * Get text from photo
2. Second Technique
    * Convert to HSV
    * Relax all bright colors from image
    * Convert image to grey
    * Apply noise removal
    * Apply Otsu thresholding
    * Apply Openning for 1 iteration
    * Apply Clossing for 2 iterations
    * Get the text from image
