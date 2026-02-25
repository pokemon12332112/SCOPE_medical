job_name='pretrain_cxr'
python ../../main_scope.py \
--data_name mimic_cxr \
--version "${job_name}" \
--task "pretrain" \
--ann_path "" \
--view_position_embed "" \
--images_dir "" \
--max_length 100 \
--is_save_checkpoint "yes" \
--is_multiview_learning "yes" \
--is_indication "yes" \
--ckpt_zoo_dir "" \
--report_style "factual_serialization" \
--pt_lr 5.0e-5 \
--epochs 50 \
--batch_size 1
