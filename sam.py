
!pip install -q segment-geospatial groundingdino-py leafmap localtileserver

import leafmap
from samgeo.common import tms_to_geotiff
from samgeo.text_sam import LangSAM

print("Initializing LangSAM model...")

sam_checkpoint_path = "/content/drive/MyDrive/Colab Files/LangSAM/hub/checkpoints/sam_vit_h_4b8939.pth"

sam = LangSAM(checkpoint=sam_checkpoint_path)

import leafmap

lat_lon = input("Enter latitude and longitude separated by a comma (e.g. 16.500454, 80.63094): ")

lat, lon = map(float, lat_lon.split(','))

m = leafmap.Map(center=[lat, lon], zoom=18, height="800px", lite_mode=True)
m.add_basemap("SATELLITE")

m

bbox = m.user_roi_bounds()
if bbox is None:
    print("No ROI selected. Using default bounding box around the map center.")
    lat, lon = m.center
    width_lon = 0.0053
    height_lat = 0.0027

    bbox = [
      lon - width_lon / 2,
      lat - height_lat / 2,
      lon + width_lon / 2,
      lat + height_lat / 2
    ]

image = "Image.tif"
tms_to_geotiff(output=image, bbox=bbox, zoom=19, source="Satellite", overwrite=True)

m.layers[-1].visible = False
m.add_raster(image, layer_name="Satellite Image")

m

# csv_input = input("Enter the required regions (trees, residential, road): ")
csv_input = "trees, residential, road"
objects_list = [obj.strip() for obj in csv_input.split(',')]

for text_prompt in objects_list:
    print(f"Processing '{text_prompt}'...")
    sam.predict(image, text_prompt, box_threshold=0.24, text_threshold=0.24)

    output_raster = f"{text_prompt.replace(' ', '_')}.tif"
    sam.show_anns(
      cmap="viridis",
      add_boxes=False,
      alpha=0.5,
      blend=False,
      output=output_raster,
      title=f"Segmentation of '{text_prompt}'"
    )