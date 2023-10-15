import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio #otherwise, streamlit overrides colors! https://discuss.streamlit.io/t/streamlit-overrides-colours-of-plotly-chart/34943 
import matplotlib.pyplot as plt
pio.templates.default = "plotly"
#from query import *
import time
import joblib
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import Draw

#special imports
from streamlit_option_menu import option_menu 
from numerize.numerize import numerize
from streamlit_shap import st_shap
import mychem
import shap
import sklearn.neighbors

st.set_page_config(page_title="Dashboard",page_icon="⚛",layout="wide")
st.header("Polyurethane Design")
st.markdown("##")

# ====== Todo
# Add save design
# Add acceleration
# Add something else...


# ===== Session Management =====
if "data" not in st.session_state:
    st.session_state["design_key"] = 0

    st.session_state["data"] = pd.read_csv("data/chemical_only_3output.csv")
    tmp = st.session_state["data"]
    input_cols = [x for x in tmp.columns if "input" in x]
    output_cols = [x for x in tmp.columns if "output" in x]
    cols = [x for x in tmp.columns if "output" in x]
    cols.extend(input_cols)
    cols_and_batch = cols.copy()
    cols_and_batch.insert(0,"batch")
    df2 = pd.DataFrame(columns=cols)
    st.session_state["blank"] = df2.copy()

    st.session_state["trial"] = pd.DataFrame(columns=cols_and_batch)
    

if "model" not in st.session_state:
    model_name = "model3output"
    with open(f"data/{model_name}.joblib", 'rb') as f:
        st.session_state["model"] = joblib.load(f)


# ===== Main Page =====
theme_plotly = 'streamlit' # None or streamlit


# Style
# with open('style.css')as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#fetch data
#result = view_all_data()
#df=pd.DataFrame(result,columns=["Policy","Expiry","Location","State","Region","Investment","Construction","BusinessType","Earthquake","Flood","Rating","id"])

 
#load data and model
df = st.session_state["data"]
df_trial = st.session_state["trial"]

input_cols = [x for x in df.columns if "input" in x]
output_cols = [x for x in df.columns if "output" in x]
cols = [x for x in df.columns if "output" in x]
cols.extend(input_cols)
cols_and_batch = cols.copy()
cols_and_batch.insert(0,"batch")
#df2 = pd.concat([df2,df])

predictor = st.session_state["model"]


#sampInput = X_test[1:2].values
#result = predictor.predict(sampInput)[0]
#x,y,z = result



#side bar
st.sidebar.image("data/Dark_Blue_Vertical.png",caption="NobleAI <> Internal")


def page_Charting():
    df["idx"] = df.index
    xchoice = st.selectbox("X-axis charting variable",
                           ['input_diol_nO', 'input_diol_nCO2', 'input_diol_nCO3', 'input_diol_nC',
       'input_diol_nCH3', 'input_iso_nC6H6', 'input_iso_nC6H12',
       'input_iso_nC', 'input_iso_nCH3', 'input_Ui', 'input_NCO/OH PP','output_Tm_ss'])
    #px.scatter(df,x="output_Tm_ss",y="output_E")

    #fig = px.box(df,x="output_E",points="all",hover_data=["idx","output_Tm_ss","input_Ui","input_NCO/OH PP"])
    fig = px.box(df,y="output_E",x=xchoice,points="all",hover_data=["idx","output_Tm_ss","input_Ui","input_NCO/OH PP"])

    #add trace
    #df2 = df.copy()
    #df2["output_E"] = df2["output_E"] + 2
    #px.box(df2,y="output_E",x=xchoice,points="all",hover_data=["idx","output_Tm_ss","input_Ui","input_NCO/OH PP"])

    fig.update_layout(showlegend=False)
    fig.update_layout(
        width=400,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""---""")
    #st.data_editor(df)
    st.markdown("""---""")

diol_inputs = ['input_diol_nO', 'input_diol_nCO2', 'input_diol_nCO3', 'input_diol_nC','input_diol_nCH3']
diol_values = {k:0 for k in diol_inputs}
diol_values['input_diol_nC'] = 1

iso_inputs = ['input_iso_nC6H6', 'input_iso_nC6H12','input_iso_nC', 'input_iso_nCH3']
iso_values = {k:0 for k in iso_inputs}
iso_values['input_iso_nC'] = 1

other_inputs = ['input_Ui', 'input_NCO/OH PP']
other_values = {'input_Ui':0, 'input_NCO/OH PP':0}

def page_LiveDesign():
    st.markdown("""
###### Polyurethanes are "modular" polymers with endless potential and variety. 
In this demo, you will have a chance to **optimize the modulus** of polyurethanes by designing the:
- diol
- isocyanate
- the type of chain extender $(U_i)$
- mixing ratio of isocynate to diol (NCO/OH) of the prepolymer
                
In addition, we provide ML-predicted characterization of IR vibrations (CO_free_W) and the soft segment melting point (Tm_ss).
""")
    total1,total2,total3=st.columns(3,gap='large')

    if "props" not in st.session_state:
        st.session_state["props"] = {
            "E":0.,
            "CO_free_W":0.,
            "Tm_ss":0.
        }
    if "smiles" not in st.session_state:
        st.session_state["smiles"] = [
            "CC",
            "O=C=NCN=C=O"
        ]

    props = st.session_state["props"]
    mcol1,mcol2,mcol3 = st.columns(3)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Design Soft Segment (diol)")
        for key in diol_inputs:
            if key.endswith("nC"):
                diol_values[key] = st.slider(key,2,10,step=1)
            else:
                diol_values[key] = st.slider(key,0,2,step=1)

    with col2:
        st.subheader("Design Hard Segment (isocyanate)")
        for key in iso_inputs:
            if key.endswith("nC"):
                iso_values[key] = st.slider(key,1,4,step=1)
            else:
                iso_values[key] = st.slider(key,0,2,step=1)

    with col3:
        st.subheader("Other Controls")
        other_values['input_Ui'] = st.slider('input_Ui',0,1,step=1)
        other_values['input_NCO/OH PP'] = st.slider('input_NCO/OH PP',2.25,6.0)


    st.session_state["smiles"][0] = mychem.generate_diol(*[v for v in diol_values.values()])
    vals = [v for v in iso_values.values()]
    vals.append(other_values["input_Ui"])
    st.session_state["smiles"][1] = mychem.generate_iso(*vals)

    # make prediction
    tmp_dict = {}
    tmp_dict.update(diol_values)
    tmp_dict.update(iso_values)
    tmp_dict.update(other_values)
    tmp_df = pd.Series(tmp_dict).to_frame().transpose()

    props["E"], props["CO_free_W"], props["Tm_ss"] = predictor.predict(tmp_df)[0]
    st.session_state["props"].update(props)

    with total1:
        st.info('E')
        st.metric(label="",value=f"{props['E']:,.0f}")

    with total2:
        st.info('CO_free_W')
        st.metric(label="",value=f"{props['CO_free_W']:,.0f}")

    with total3:
        st.info('Tm_ss')
        st.metric(label="",value=f"{props['Tm_ss']:,.0f}")

    with mcol1:
        mol1 = st.session_state["smiles"][0]
        mol1 = Chem.MolFromSmiles(mol1)

        d1 = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(300,200)
        d1.DrawMolecule(mol1)
        d1.FinishDrawing()
        svg1 = d1.GetDrawingText().replace('svg:','')
        st.image(svg1)
    with mcol2:
        mol2 = st.session_state["smiles"][1]
        mol2 = Chem.MolFromSmiles(mol2)

        d2 = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(300,200)
        d2.DrawMolecule(mol2)
        d2.FinishDrawing()
        svg2 = d2.GetDrawingText().replace('svg:','')
        st.image(svg2)
#graphs

def page_Design():
    st.markdown("###### On this page you have a chance to more systematically explore designs, save your results, and view them.")
    st.markdown("## Charting")
    charting = st.expander("## Charting")

    st.markdown("## New Designs")
    with st.expander("New Designs",expanded=True):
        # page_cols = st.columns( len(output_cols) )
        # for ii,pg in enumerate(page_cols):
        #     with pg:
        #         st.info(output_cols[ii])
        #         st.metric(label="",value=f"{23.12:,.1f}")

        defaults = {input_field:st.column_config.NumberColumn(input_field,default=0.) for input_field in input_cols
        }
        c1,c2 = st.columns(2)
        with c1:
            edited = st.data_editor(st.session_state["blank"][input_cols].copy(),
                                column_config=defaults,
                                disabled=output_cols,
                                num_rows="dynamic",
                                key=st.session_state["design_key"])
        
        with c2:
            if len(edited) > 0:
                outputs = predictor.predict(edited[input_cols])
                new_data = pd.DataFrame(outputs,columns=output_cols)
                st.dataframe( new_data )

        if len(edited) > 0:
            new_designs = pd.DataFrame( np.hstack([outputs,edited.values.copy()]), columns=cols)
            new_designs["batch"] = st.session_state["design_key"]
            new_designs = new_designs[cols_and_batch].copy()

            #st.dataframe(new_designs)

        st.button("Save designs",on_click=lambda: save(edited))

    st.markdown("## Saved Designs")
    with st.expander("Saved Designs"):
        st.write(st.session_state["trial"])

    #save
    def save(edited):
        if len(edited) > 0:
            #st.session_state["trial"] = pd.concat([st.session_state["trial"],new_designs])
            st.session_state["trial"] = pd.concat( [st.session_state["trial"],new_designs],ignore_index=True )
            #edited.drop(edited.index,inplace=True)
            st.session_state["design_key"] += 1


    with charting:
        tmp_c1, tmp_c2 = st.columns(2)
        with tmp_c1:
            xax = st.selectbox("X-axis",cols)
        with tmp_c2:
            yax = st.selectbox("Y-axis",cols)
        tmp = st.session_state["trial"]
        fig = px.scatter(tmp,x=xax,y=yax,color="batch",hover_data=[tmp.index])
        fig.update_layout(
            width=400,
            height=400,
            xaxis_title_font_size=20,
            xaxis_tickfont_size = 16,
            yaxis_title_font_size=20,
            yaxis_tickfont_size = 16,
            hoverlabel_font_size=16,
            legend_font_size=16,
        )
        fig.update_traces(marker={'size':12,'opacity':0.5})
        if len(edited) > 0:
            fig.add_trace( px.scatter(new_designs,x=xax,y=yax,
                                      color = [-1]*len(new_designs),
                                      color_discrete_sequence=["#0068c9"]).data[0])
            fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(marker={'size':12})

        st.plotly_chart(fig)
        

    # with st.form("Data form"):
    #     saved = st.form_submit_button("Save")
    #     if saved:
    #         st.session_state["trial"] = edited.copy()
    #         st.experimental_rerun()

    # with st.form("Data form"):
    #     submitted = st.form_submit_button("Predict")

    #     #st.write(st.session_state["trial"])
    #     if submitted:
    #         if len(edited) > 0:
    #             inputs = edited.values
    #             outputs = predictor.predict(edited[input_cols])
    #             edited[output_cols] = outputs
    #         #new_df = pd.DataFrame( np.hstack([outputs,inputs]), columns=cols )
    #         st.write(edited)
    #         st.session_state["trial"] = edited
    #         #st.session_state["trial"] = pd.concat([st.session_state["trial"]])
    #         #st.session_state["trial"] = edited.copy()
    #         st.experimental_rerun()

    #     saved = st.form_submit_button("Save")
    #     if saved:
    #         st.session_state["trial"] = edited.copy()
    #         st.experimental_rerun()

    # st.dataframe(st.session_state["trial"])
    #st.experimental_rerun()
    #st.session_state["trial"] = edited.copy()
    #st.dataframe(st.session_state["trial"])

    #df_trial = df
    #df_trial[output_cols] = predictor.predict(df[input_cols])
    #st.dataframe(df_trial)
    #st.session_state["trial"] = df_trial

def page_Acquire():
    st.markdown("###### Here we simulate using the ML model to propose and guide experiments. After acquiring true experimental data, we can then compare to the ML predictions.")
    st.markdown("In the simulation below, we begin with collecting 5 true experimental data at random, then using a model to guide selection of top experimental candidates.")
    st.markdown("How long does it take for the baseline model to find the same high-performing (E) material? This simulates the effective acceleration from using model-guided design.")
    # Data structures:
    # Existing data --> can make it just a list of indices
    # Competing random acquisition --> can make it just a list of indices
    # Proposed data + checkboxes? 
    # idea: interactive
    import random

    initial_amount = 5
    batch_size = 5

    def remove_from_remaining(pick,options):
        return [x for x in options if x not in pick]

    def reset_data():
        st.session_state.curated_remaining = list(df.index)
        st.session_state.baseline_remaining = list(df.index)

        st.session_state.curated = random.choices(st.session_state.curated_remaining, k=initial_amount)
        st.session_state.curated_remaining = remove_from_remaining(st.session_state.curated,st.session_state.curated_remaining)
        st.session_state.baseline = st.session_state.curated.copy()
        st.session_state.baseline_remaining = remove_from_remaining(st.session_state.baseline,st.session_state.baseline_remaining)

    def acquire():
        random_add = random.choices(st.session_state.baseline_remaining, k = batch_size)
        st.session_state.baseline_reamaining = remove_from_remaining(random_add, st.session_state.baseline_remaining)
        st.session_state.baseline.extend(random_add)

        #prediction guidance
        predictions = predictor.predict( df.iloc[st.session_state.curated_remaining][input_cols] )
        arg_inds = np.argsort( -predictions[:,0] ) #sort by E, first column
        ML_add = np.array(st.session_state.curated_remaining)[arg_inds[:5].tolist()]
        st.session_state.curated_remaining = remove_from_remaining(ML_add, st.session_state.curated_remaining)
        st.session_state.curated.extend(ML_add)


    if "curated" not in st.session_state:
        st.session_state["curated"] = []
        st.session_state["curated_remaining"] = []
        st.session_state["baseline"]  = []
        st.session_state["baseline_remaining"] = []
        reset_data()

    # Button: Reset
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        reset_button = st.button("Reset")
        if reset_button:
            reset_data()
    with bcol2:
        acquire_button = st.button("Acquire Exp. Data & Retrain")
        if acquire_button:
            acquire()

    st.markdown("---")
    st.markdown("##### Results")
    # Show Current Data via plot
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        fig,ax = plt.subplots(figsize=(3,3),dpi=200)
        ax.scatter( df.iloc[st.session_state.baseline].output_Tm_ss, 
                df.iloc[st.session_state.baseline].output_E, 
                label="baseline", alpha=0.5 )
        ax.scatter( df.iloc[st.session_state.curated].output_Tm_ss, 
                df.iloc[st.session_state.curated].output_E, 
                label ="ML acquired", alpha=0.5 )
        ax.set_xlabel("Tm_ss (brittleness)")
        ax.set_ylabel("E (stiffness)")
        ax.legend()
        st.pyplot(fig)

    with vcol2:
        fig2,ax2 = plt.subplots(figsize=(3,3),dpi=200)
        ax2.plot( df.iloc[st.session_state.baseline].output_E.values,
                 label ="baseline", alpha=0.5 )
        ax2.plot( df.iloc[st.session_state.curated].output_E.values,
                 label ="ML acquired", alpha=0.5 )
        ax2.set_xlabel("iterations")
        ax2.set_ylabel("E (stiffness)")
        ax2.legend()
        st.pyplot(fig2)

    # Show Proposals: Expander
    # OPTIONAL to make it editable

    # Show Old Acquisitions: Expander
    # OPTIONAL

    # Button: acquire and retrain


    # Plot results: r2, acceleration rate?




def page_Analyses():
    # Show other benchmarks from my study
    # See another explainability tutorial: https://app-california-app.streamlit.app/
    #   look at instructions, interactively explore, etc.
    # A typical SHAP calculation is itself ~20000 model runs (!)
    st.markdown("""
    ### Benchmarks
###### We have done additional analyses to benchmark and prove the efficacy of our models.
""")
    c1,c2 = st.columns(2)
    with c1:
        st.image("data/baseline.png")
        st.markdown('In these calculations, we verified that a science-inspired ML architecture mimicking multi-scale structure-property relationships outperforms a naive, direct black box ML model. In addition, our architecture gives us physical characterization "for free" and only uses physically controllable variables as inputs.')

    with c2:
        st.image("data/r2_vs_training_volume.png")
        st.markdown('We find that our ML models perform quite well even with just \~15 data points, with steady improvement as we approach 30+ data points. Note that even an (optimized) Box-Behnken DOE requires \~200 experiments. Our data efficiency means we potentially only need 10~30% of the typical DOE data to characterize the design space.')
    

    st.markdown("""---""")
    st.markdown("""
### Full chemical space sweep
""")
    design_base = {
        "diol_nO":0,
        "diol_nCO2":0,
        "diol_nCO3":1,
        "diol_nC":6,
        "diol_nCH3":0
    }
    import time
    start = time.time()
    designs = []
    for nC6H6 in [0,1,2]:
        for nC6H12 in [0,1,2]:
            for nC in [2,4,6]:
                for nCH3 in [0,1,2]:
                    for Ui in [0,1]:
                        for NCO_OH in [2,3,4,5,6]:
                            for nC_diol in [2,4,6,8]:
                                for nCH3_diol in [0,1,2]:
                                    for nCO3_diol in [0,1,2]:
                                        for nCO2_diol in [0,1,2]:
                                            for nO_diol in [0,1,2]:
                                                design = design_base.copy()
                                                design.update({"iso_nC6H6": nC6H6,
                                                            "iso_nC6H12": nC6H12,
                                                            "iso_nC": nC,
                                                            "iso_nCH3": nCH3,
                                                            "Ui": Ui,
                                                            "NCO/OH PP":NCO_OH,
                                                            "diol_nC":nC_diol,
                                                            "diol_nCH3":nCH3_diol,
                                                            "diol_nO":nO_diol,
                                                            "diol_nCO2":nCO2_diol,
                                                            "diol_nCO3":nCO3_diol,
                                                            })
                                                designs.append(design)
    new_designs = pd.DataFrame(designs)
    new_outputs = predictor.predict(new_designs.values)

    # xchoice = st.selectbox("X-axis charting variable",
    #                        ['input_diol_nO', 'input_diol_nCO2', 'input_diol_nCO3', 'input_diol_nC',
    #    'input_diol_nCH3', 'input_iso_nC6H6', 'input_iso_nC6H12',
    #    'input_iso_nC', 'input_iso_nCH3', 'input_Ui', 'input_NCO/OH PP','output_Tm_ss'])
    
    new_df = pd.DataFrame( np.hstack([new_outputs,new_designs.values]), columns=cols )

    #fig = px.box(new_df,y="output_E",x=xchoice,hover_data=[new_df.index,"output_Tm_ss","input_Ui","input_NCO/OH PP"])

    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown(f"**{time.time()-start:.02}**s to calculate for: {int(len(designs)/5)} chemical combinations, {len(designs)} total evaluations")
        fig = px.box(new_df,y="output_E")
        fig.update_layout(showlegend=False)
        fig.update_layout(
            width=400,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=False)
    with tc2:
        xchoice = st.selectbox("X-axis charting variable",['output_Tm_ss', 'output_CO_free_W'])
        fig = px.scatter(new_df,x=xchoice,y="output_E")
        fig.update_layout(showlegend=False)
        fig.update_layout(
            width=400,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=False)


    st.write("Top 10 designs:")
    new_df.sort_values("output_E",ascending=False,inplace=True)
    top10 = new_df.head(100).sample(n=10).copy()
    top10.sort_values("output_E",ascending=False,inplace=True)
    st.write(top10)

    
    st.markdown("""---""")
    st.markdown("""### Overall SHAP interpretability scores""")
    start = time.time()
    st.markdown(f"(Note this may take a few seconds since it involves ~{ int(np.round(len(input_cols)*2.048,0)*1000)*len(df) } inferences.)")
    X, y = df[input_cols], df[output_cols]
    def f(x):
        return predictor.predict(x)
    explainer = shap.KernelExplainer(f,X)
    shap_values = explainer(X)[:,:,0]
    st.markdown(f"**{time.time()-start:.03}**s to evaluate shap")
    st_shap(shap.plots.beeswarm(shap_values))
    # st_shap(shap.plots.waterfall(shap_values[0]))


def page_Bonus():
    st.markdown("""
## Paid Features
- Advanced acquisition functions
- Advanced optimization
- Advanced analysis
    - Dimensionality reduction
- Automated exploration of design space
- Customized data views
- Data annotation
""")


def sideBar():
    with st.sidebar:
        selected=option_menu(
            menu_title="Main Menu",
            options=["Live Design","Design","ML-Guided Acceleration","Analyses", "Bonus"], #Charting / info-circle / graph-up
            icons=["eye","clipboard2-data","upload","graph-up", "infinity"], #house, icons from: https://icons.getbootstrap.com/
            menu_icon="cast",
            default_index=0
        )
    if selected == "Design":
        page_Design()
    if selected=="Live Design":
        #st.subheader(f"Page: {selected}")
        page_LiveDesign()
        # graphs()
    if selected == "Charting":
        page_Charting()
    if selected=="Analyses":
        #st.subheader(f"Page: {selected}")
        page_Analyses()
        # graphs()
    if selected == "Bonus":
        page_Bonus()
    if selected =="ML-Guided Acceleration":
        page_Acquire()

sideBar()



#theme
hide_st_style=""" 

<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""



