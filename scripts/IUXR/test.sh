job_name='test_iu_xray'
python ../../main_scope.py \
--data_name iu_xray \
--version "${job_name}" \
--task "test" \
--ann_path "" \
--view_position_embed "" \
--images_dir "" \
--max_length 100 \
--is_save_checkpoint "yes" \
--is_multiview_learning "yes" \
--is_prior_scan "yes" \
--using_mpc_loss "no" \
--is_indication "yes" \
--is_prior_report "yes" \
--ckpt_zoo_dir "" \
--test_ckpt_path "" \
--cvt2distilgpt2_path "" \
--num_workers 8 \
--batch_size 32