### Raw to synthetic domain transformation for OCR images. 

The project is structured as follows.

#### Codebase structure
```
dataset # training data
images # save generated images here
saved_model # save weights and model configurations here
test_imgs # images for testing/generating target domain images

.py files #python scripts for training and inference
```
#### Dataset directory strucuture:

```
dataset/
	org2synth/
		Domain_A/
			# all real world OCR data
		Domain_B/
			# all synthetic OCR data
```
This is divided in the following file structure for the training regiment.
```
dataset/
	org2synth/
			train_A/
				# real world images for training
			train_B/
				# synthetic images for training
			test_A/
				# around 300 samples real world images
			test_B/
				# around 300 samples synthetic images
```
