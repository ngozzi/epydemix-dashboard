import streamlit as st

class StepperBar:
    def __init__(self, steps):
        self.steps = steps
        self.current_step = 0

    def set_current_step(self, step: int):
        if 0 <= step < len(self.steps):
            self.current_step = step
        else:
            raise ValueError("Step index out of range")

    def display(self):
        # Simple stepper using Streamlit components
        cols = st.columns(len(self.steps))
        
        for i, step in enumerate(self.steps):
            with cols[i]:
                if i < self.current_step:
                    # Completed step
                    st.markdown("""
                    <div style='text-align:center;'>
                        <div style='width:26px;height:26px;border-radius:50%;background:rgba(148,163,184,0.6);display:inline-block;margin-bottom:8px;'></div>
                        <div style='color:#cbd5e1;font-size:13px;'>{}</div>
                    </div>
                    """.format(step), unsafe_allow_html=True)
                elif i == self.current_step:
                    # Current step
                    st.markdown("""
                    <div style='text-align:center;'>
                        <div style='width:26px;height:26px;border-radius:50%;background:#60f0d8;display:inline-block;margin-bottom:8px;'></div>
                        <div style='color:#e5e7eb;font-size:13px;'>{}</div>
                    </div>
                    """.format(step), unsafe_allow_html=True)
                else:
                    # Future step
                    st.markdown("""
                    <div style='text-align:center;'>
                        <div style='width:26px;height:26px;border-radius:50%;background:rgba(148,163,184,0.25);display:inline-block;margin-bottom:8px;'></div>
                        <div style='color:#94a3b8;font-size:13px;'>{}</div>
                    </div>
                    """.format(step), unsafe_allow_html=True)
        
        # Add connecting lines using CSS
        st.markdown("""
        <style>
        .stepper-lines {
            position: relative;
            height: 2px;
            background: rgba(148,163,184,0.25);
            margin: -15px 0 20px 0;
            z-index: 1;
        }
        </style>
        <div class="stepper-lines"></div>
        """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Stepper", layout="wide", initial_sidebar_state="collapsed")

    steps = ["Model", "Geography", "Parameters", "Settings", "Run"]

    # Persist current step
    st.session_state.setdefault("step", 0)

    # Visual stepper display
    stepper = StepperBar(steps)
    stepper.set_current_step(st.session_state["step"])
    stepper.display()

    # Content placeholder for the current step
    st.write(f"Step {st.session_state['step'] + 1} of {len(steps)} — {steps[st.session_state['step']]}")

    # Navigation buttons
    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        if st.button("Back", disabled=st.session_state["step"] == 0, use_container_width=True):
            st.session_state["step"] -= 1
            st.rerun()
    with col_next:
        if st.button("Next", disabled=st.session_state["step"] == len(steps) - 1, use_container_width=True):
            st.session_state["step"] += 1
            st.rerun()


if __name__ == "__main__":
    main()