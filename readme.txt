cd VITON-HD

python test.py --name viton_test --dataset_dir datasets --dataset_mode test --dataset_list test_pairs.txt --checkpoint_dir checkpoints --save_dir results

streamlit run app.py