
from util import get_artifacts
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Visit with Us: Wellness Purchase Propensity", page_icon="🛍️", layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding-top: 5rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
    </style>
    """, unsafe_allow_html=True)

st.image("assets/banner.png", width='content')
st.header("Visit with Us: Wellness Purchase Propensity")

def render_form(schema):
    user_inputs = {}
    with st.form("main_form"):
        col1, col2 = st.columns(2)        
        for idx, field in enumerate(schema["inputs"]):
            # alternating columns
            target_col = col1 if idx % 2 == 0 else col2    
            # layout        
            with target_col:
                name = field["name"]
                label = field["label"]
                ftype = field["type"]

                # SELECT INPUT
                if ftype == "select":
                    options = field["options"]
                    if field.get("binary"):
                        # checkbox
                        is_checked = st.checkbox(label, key=f"chk_{name}")
                        # value mapping from the options list [No, Yes | 0, 1]
                        user_inputs[name] = options[1] if is_checked else options[0]
                    else:
                        # dropdown
                        if isinstance(options, dict):
                            choice = st.selectbox(label, options=list(options.keys()), key=f"sel_{name}")
                            user_inputs[name] = options[choice]
                        else:
                            user_inputs[name] = st.selectbox(label, options=options, key=f"sel_{name}")

                # NUMBER INPUT
                elif ftype == "number":
                    is_int = field.get("is_int", False)
                    caster = int if is_int else float # conversion function based on is_int
                    user_inputs[name] = st.number_input(
                        label, 
                        min_value=caster(field["min"]),
                        max_value=caster(field["max"]),
                        step=caster(field.get("step", 1)),
                        key=f"num_{name}"
                    )

                # TEXT INPUT
                elif ftype == "text":
                    user_inputs[name] = st.text_input(label, key=f"txt_{name}")
        # eo: for
        submit = st.form_submit_button("Submit")        
    return user_inputs, submit

# exec
model, schema = get_artifacts()
features = [
    f["name"] for f in schema["inputs"]
]
user_input, submit = render_form(schema)
if submit:
    if not all(features):
        st.error("Please fill in all the required fields")
    else:
        df = pd.DataFrame([user_input])
        proba = model.predict_proba(df)[:, 1]
        pred = (proba >= 0.5).astype(int)
        st.divider()
        st.subheader("Results:")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if pred == 1:
                st.success("### YES")
                st.metric("Confidence", f"{round(proba * 100, 2)}%")
            else:
                st.error("### NO")
                st.metric("Confidence", f"{round((1 - proba) * 100, 2)}%")
        with res_col2:
            st.write("**Confidence Level:**")
            st.progress(proba)
            st.caption("The progress bar indicates the probability of a 'Yes' outcome.")
        st.balloons()
