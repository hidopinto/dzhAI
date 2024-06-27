from backbone.re_clean_data import main as reclean_main
from backbone.predict_mortality import main as pred_main
from backbone.eval_model import main as eval_main
from backbone.run_shap import main as shap_main


def main():
    reclean_main()
    pred_main()
    eval_main()
    shap_main()


if __name__ == '__main__':
    main()
