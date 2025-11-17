DATA_DIR=/opt/gpudata/cxr-derived

docker run --rm -v "$DATA_DIR":/data chexpert-labeler:latest python label.py --reports_path /data/mimic-findings.csv --output_path /data/mimic-findings-labels.csv --verbose
docker run --rm -v "$DATA_DIR":/data chexpert-labeler:latest python label.py --reports_path /data/mimic-impression.csv --output_path /data/mimic-impression-labels.csv --verbose
docker run --rm -v "$DATA_DIR":/data chexpert-labeler:latest python label.py --reports_path /data/chexpertplus-findings.csv --output_path /data/chexpertplus-findings-labels.csv --verbose
docker run --rm -v "$DATA_DIR":/data chexpert-labeler:latest python label.py --reports_path /data/chexpertplus-impression.csv --output_path /data/chexpertplus-impression-labels.csv --verbose

# next step overwrites labeler output, backup labels
cp $DATA_DIR/mimic-findings-labels.csv $DATA_DIR/mimic-findings-labels.csv.BAK
cp $DATA_DIR/mimic-impression-labels.csv $DATA_DIR/mimic-impression-labels.csv.BAK
cp $DATA_DIR/chexpertplus-findings-labels.csv $DATA_DIR/chexpertplus-findings-labels.csv.BAK
cp $DATA_DIR/chexpertplus-impression-labels.csv $DATA_DIR/chexpertplus-impression-labels.csv.BAK
