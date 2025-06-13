with vis_col:
    st.subheader("👁️ Molecule Viewer")
    current_smiles_for_vis = st.session_state.get('smiles_input', DEFAULT_SMILES)
    if current_smiles_for_vis:
        if not RDKIT_DRAW_ENABLED:
            st.warning("Molecule rendering unavailable. Displaying SMILES instead.")
            st.code(current_smiles_for_vis)
        else:
            mol_image = smiles_to_image(current_smiles_for_vis)
            if mol_image:
                st.image(mol_image, caption=f"Structure: {current_smiles_for_vis}", use_column_width=True)
            else:
                st.warning("Molecule image unavailable. Displaying SMILES instead.")
                st.code(current_smiles_for_vis)
    else:
        st.info("Enter a SMILES string or search PubChem to see the molecule structure.")

    if pubchem_name and pubchem_url: 
        st.markdown(f"🔗 [View **{pubchem_name}** on PubChem]({pubchem_url})", unsafe_allow_html=True)
