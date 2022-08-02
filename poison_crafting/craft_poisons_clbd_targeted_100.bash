for idx in {0..99} 
do
    echo Crafting batch $idx...
    python poison_crafting/craft_poisons_clbd_targeted.py --setup_idx $idx --poisons_path poison_examples/clbd_targeted_CT
done