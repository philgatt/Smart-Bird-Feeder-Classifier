# Bird Feeder Species Classifier

A classifier for images and videos from a fixed bird-feeder camera. It predicts the
species visible in a frame, or `No Bird` for an empty feeder.

## Supported species

The primary model predicts these 11 classes:

| German | English |
| --- | --- |
| Blaumeise | Blue tit |
| Eichelhäher | Eurasian jay |
| Feldsperling | Eurasian tree sparrow |
| Kleiber | Eurasian nuthatch |
| Kohlmeise | Great tit |
| Rotkehlchen | European robin |
| Specht | Woodpecker |
| Sumpfmeise | Marsh tit |
| Tannenmeise | Coal tit |
| Taube | Common wood pigeon |
| Kein Vogel (`No Bird`) | No bird |

An extended model additionally predicts rare species using synthetic images: Amsel
(common blackbird), Grünfink (European greenfinch), and Stieglitz (European goldfinch).

## Accuracy

The first three models were tested on the **same 426 held-out real feeder-camera
images**, so their scores are directly comparable.

| Model | Training data | Accuracy on held-out real images | Meaning |
| --- | --- | --- | --- |
| **Real-only ConvNeXt-Tiny** | Real images | **94.8%** | Best checkpoint for the established 11 classes. |
| Combined ConvNeXt-Tiny | Real and synthetic images | 93.4% | Synthetic augmentation did not improve the well-covered classes. |
| Synthetic-only ConvNeXt-Tiny | Synthetic images | 60.1% | Useful for rare-class extension, but not a replacement for real images. |
| Extended ConvNeXt-Tiny | Real core classes; synthetic common blackbird, European greenfinch, and European goldfinch | No single reliable score yet | Those rare classes have only 7, 30, and 5 real test images. |

Synthetic data is used primarily to add rare species. No held-out real test image was
used to generate the synthetic training data.

### Example predictions

These are four correctly classified images from the held-out real test set. Certainty
is the model's probability for its predicted class.

| Test image | Prediction | Certainty |
| --- | --- | --- |
| <img src="docs/images/test-predictions/blue-tit.jpg" alt="Real test image of a blue tit" width="224"> | Blue tit (Blaumeise) | 99.95% |
| <img src="docs/images/test-predictions/eurasian-jay.jpg" alt="Real test image of a Eurasian jay" width="224"> | Eurasian jay (Eichelhäher) | 99.95% |
| <img src="docs/images/test-predictions/european-robin.jpg" alt="Real test image of a European robin" width="224"> | European robin (Rotkehlchen) | 99.95% |
| <img src="docs/images/test-predictions/common-wood-pigeon.jpg" alt="Real test image of a common wood pigeon" width="224"> | Common wood pigeon (Taube) | 99.90% |

## Train

```powershell
conda create -n bird-classifier python=3.11 pip
conda activate bird-classifier
pip install -e ".[dev]"
```

Train the primary 11-class model:

```powershell
python -u scripts/train_classifier.py --data-source real --architecture convnext-tiny --class-set core --epochs 30 --batch-size 32
```

Train the extended model:

```powershell
python -u scripts/train_classifier.py --data-source combined --architecture convnext-tiny --class-set extended --epochs 30 --batch-size 32
```

## Predict a folder

```powershell
python scripts/batch_predict_media_folder.py F:/BirdCamera/recordings --model F:/path/to/model.pt --recursive
```

The prediction report is saved as `bird_detection_report.json` in the input folder.
