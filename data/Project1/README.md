# Hyperspectral Data Documentation

## 1. Data Description
- **Dataset Name**: Proximal Hyperspectral Images of Agricultural Plants
- **Version**: 1.0
- **Description**: This dataset contains proximal hyperspectral images of various agricultural plants (canola, soybean, sugar beet) and weeds (kochia, ragweed, redroot pigweed, waterhemp). The images were collected as part of a study on plant species discrimination.
- **Data Format**: Numpy Array (.npy)
- **Volume**: 160 images in total, distributed across different species

## 2. Data Structure
- **Image Format**:
  - Spectral Range: 400-1000 nm
  - Type: Calibrated hyperspectral images
  - Organization: Each .npy file contains a complete hyperspectral image

- **Directory Structure**:
  ```
  proximal_hyperspectral_image/
  ├── canola/ (20 images)
  ├── soybean/ (20 images)
  ├── sugarbeet/ (20 images)
  ├── kochia/ (20 images)
  ├── ragweed/ (20 images)
  ├── redroot_pigweed/ (40 images)
  └── waterhemp/ (20 images)
  ```

## 3. Data Collection Methodology
- **Equipment**:
  - Sensor: Specim FX10 hyperspectral sensor
  - Platform: SPECIM's LabScanner system
  - Software: Lumo Scanner
- **Conditions**: Images collected under controlled halogen light source
- **Calibration**: Images calibrated using white and dark reference images
- **Acquisition Protocol**:
  - 4 plants per image for most species
  - 1 plant per image for redroot pigweed

## 4. Terms of Use
- **License**: This dataset is distributed under the Creative Commons Attribution 4.0 International (CC-BY 4.0) license
- **Funding**: USDA: 58-6064-8-023

## 5. Limitations and Potential Biases
- **Controlled Conditions**: Images were taken under laboratory conditions, which may differ from field conditions
- **Representativeness**: The limited number of images per species (20-40) may affect model generalization

## 6. How to Use the Data
- **Prerequisites**:
  - Python 3.x
  - NumPy
  - Jupyter Notebook (for support tools)
- **Potential Applications**:
  - Development of classification/identification models
  - Spectroscopy studies
  - Development of three-dimensional data models

## 7. References and Attribution
To cite this dataset, please use:
```
USDA Proximal Hyperspectral Image Dataset (2024)
Funding: USDA: 58-6064-8-023
```

## 8. Contact
For any questions regarding this dataset, please contact [to be completed]

## 9. Additional Tools
Jupyter notebooks are provided for:
- Data augmentation
- Region of interest selection
- Spectral preprocessing 