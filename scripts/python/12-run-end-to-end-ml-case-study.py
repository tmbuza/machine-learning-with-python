"""Run the complete MLPY-012 retention-model case study."""

from hashlib import sha256
import json
from pathlib import Path
import platform

from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results" / "tables"
REPORT_PATH = ROOT / "results" / "reports" / "12-project-handoff.md"
FIGURE_PATH = ROOT / "results" / "figures" / "12-end-to-end-ml-case-study.png"
ARTIFACT_DIR = ROOT / "artifacts"
FEATURES = ["tenure_months", "monthly_charge", "support_contacts", "usage_hours", "plan", "region"]
NUMERIC, CATEGORICAL = FEATURES[:4], FEATURES[4:]


def create_data():
    rng = np.random.default_rng(SEED); rows=[]; count=430
    ids=[f"C{n:04d}" for n in range(1,count+1)]; plans=rng.choice(["Basic","Standard","Premium"],count,p=[.42,.38,.2]); regions=rng.choice(["North","South","East","West"],count); groups=rng.choice(["Group A","Group B","Group C"],count,p=[.52,.31,.17]); risk=rng.normal(0,.75,count); base=rng.normal(43,9,count)
    dates=pd.date_range("2026-01-31",periods=8,freq="ME")
    for month,date in enumerate(dates):
        for i,cid in enumerate(ids):
            tenure=int(rng.integers(2,70)+month); contacts=int(rng.poisson(1.1+.45*max(risk[i],0))); usage=base[i]+{"Basic":-7,"Standard":1,"Premium":9}[plans[i]]+month+rng.normal(0,5); charge={"Basic":35,"Standard":65,"Premium":95}[plans[i]]+rng.normal(0,5); logit=-1.4+risk[i]+.55*contacts-.02*tenure-.043*usage+.08*month+.38*(plans[i]=="Basic")+.18*(groups[i]=="Group C"); probability=1/(1+np.exp(-logit)); target=int(rng.random()<probability) if month<7 else np.nan
            rows.append({"customer_id":cid,"snapshot_date":date,"audit_group":groups[i],"tenure_months":tenure,"monthly_charge":round(float(charge),2),"support_contacts":contacts,"usage_hours":round(max(float(usage),0),2),"plan":plans[i],"region":regions[i],"churned_next_month":target})
    data=pd.DataFrame(rows); data.loc[rng.choice(data.index,60,replace=False),"monthly_charge"]=np.nan; return data,dates


def prep():
    return ColumnTransformer([("numeric",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),NUMERIC),("categorical",make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore")),CATEGORICAL)])


def candidates():
    return {"Prior baseline":make_pipeline(prep(),DummyClassifier(strategy="prior")),"Logistic regression":make_pipeline(prep(),LogisticRegression(class_weight="balanced",max_iter=1000,random_state=SEED)),"Shallow tree":make_pipeline(prep(),DecisionTreeClassifier(class_weight="balanced",max_depth=4,min_samples_leaf=25,random_state=SEED))}


def main():
    for p in [ROOT/"data"/"processed",ROOT/"data"/"scoring",TABLE_DIR,REPORT_PATH.parent,FIGURE_PATH.parent,ARTIFACT_DIR]:p.mkdir(parents=True,exist_ok=True)
    data,dates=create_data(); data.to_csv(ROOT/"data"/"processed"/"12-customer-snapshots.csv",index=False,date_format="%Y-%m-%d")
    train=data[data.snapshot_date.isin(dates[:5])]; validation=data[data.snapshot_date==dates[5]]; test=data[data.snapshot_date==dates[6]].copy(); scoring=data[data.snapshot_date==dates[7]].drop(columns="churned_next_month").copy(); scoring.to_csv(ROOT/"data"/"scoring"/"12-scoring-batch.csv",index=False,date_format="%Y-%m-%d")
    comparison=[]
    for name,model in candidates().items():
        model.fit(train[FEATURES],train.churned_next_month.astype(int)); prob=model.predict_proba(validation[FEATURES])[:,1]; pred=(prob>=.5).astype(int); comparison.append({"model":name,"validation_roc_auc":roc_auc_score(validation.churned_next_month,prob),"validation_balanced_accuracy":balanced_accuracy_score(validation.churned_next_month,pred),"validation_brier_score":brier_score_loss(validation.churned_next_month,prob)})
    comparison=pd.DataFrame(comparison); comparison.to_csv(TABLE_DIR/"12-model-comparison.csv",index=False)
    selected=candidates()["Logistic regression"]; selected.fit(train[FEATURES],train.churned_next_month.astype(int)); val_prob=selected.predict_proba(validation[FEATURES])[:,1]
    thresholds=[]
    for threshold in np.arange(.1,.91,.02):
        pred=(val_prob>=threshold).astype(int); thresholds.append({"threshold":threshold,"precision":precision_score(validation.churned_next_month,pred,zero_division=0),"recall":recall_score(validation.churned_next_month,pred,zero_division=0),"f1":f1_score(validation.churned_next_month,pred,zero_division=0)})
    thresholds=pd.DataFrame(thresholds); threshold=float(thresholds.loc[thresholds.f1.idxmax(),"threshold"]); thresholds.to_csv(TABLE_DIR/"12-threshold-analysis.csv",index=False)
    development=pd.concat([train,validation]); selected.fit(development[FEATURES],development.churned_next_month.astype(int)); test_prob=selected.predict_proba(test[FEATURES])[:,1]; test_pred=(test_prob>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(test.churned_next_month,test_pred).ravel()
    metrics=pd.DataFrame([{"selected_model":"Logistic regression","threshold":threshold,"roc_auc":roc_auc_score(test.churned_next_month,test_prob),"average_precision":average_precision_score(test.churned_next_month,test_prob),"accuracy":accuracy_score(test.churned_next_month,test_pred),"balanced_accuracy":balanced_accuracy_score(test.churned_next_month,test_pred),"precision":precision_score(test.churned_next_month,test_pred,zero_division=0),"recall":recall_score(test.churned_next_month,test_pred,zero_division=0),"f1":f1_score(test.churned_next_month,test_pred,zero_division=0),"brier_score":brier_score_loss(test.churned_next_month,test_prob),"true_negative":tn,"false_positive":fp,"false_negative":fn,"true_positive":tp}]); metrics.to_csv(TABLE_DIR/"12-final-test-metrics.csv",index=False)
    imp=permutation_importance(selected,test[FEATURES],test.churned_next_month,scoring="roc_auc",n_repeats=15,random_state=SEED); importance=pd.DataFrame({"feature":FEATURES,"mean_roc_auc_decrease":imp.importances_mean,"sd_roc_auc_decrease":imp.importances_std}).sort_values("mean_roc_auc_decrease",ascending=False); importance.to_csv(TABLE_DIR/"12-feature-importance.csv",index=False)
    test["probability"]=test_prob; test["prediction"]=test_pred; group_rows=[]
    for group,frame in test.groupby("audit_group"):
        group_rows.append({"audit_group":group,"rows":len(frame),"positive_cases":int(frame.churned_next_month.sum()),"selection_rate":frame.prediction.mean(),"recall":recall_score(frame.churned_next_month,frame.prediction,zero_division=0),"precision":precision_score(frame.churned_next_month,frame.prediction,zero_division=0)})
    groups=pd.DataFrame(group_rows); groups.to_csv(TABLE_DIR/"12-subgroup-performance.csv",index=False)
    artifact=ARTIFACT_DIR/"12-retention-pipeline.joblib"; dump(selected,artifact); digest=sha256(artifact.read_bytes()).hexdigest()
    schema={"schema_version":"1.0.0","required_features":FEATURES,"identifier_fields":["customer_id","snapshot_date"],"audit_only_fields":["audit_group"],"target":"churned_next_month"}; (ARTIFACT_DIR/"12-feature-schema.json").write_text(json.dumps(schema,indent=2)+"\n")
    metadata={"model_version":"1.0.0","selected_model":"Logistic regression","decision_threshold":threshold,"training_end":dates[5].date().isoformat(),"test_period":dates[6].date().isoformat(),"test_roc_auc":float(metrics.iloc[0].roc_auc),"artifact_sha256":digest,"python_version":platform.python_version(),"scikit_learn_version":sklearn.__version__}; (ARTIFACT_DIR/"12-model-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    score_prob=selected.predict_proba(scoring[FEATURES])[:,1]; out=scoring[["customer_id","snapshot_date"]].copy(); out["prediction_id"]=[sha256(f"{cid}|{date.date()}|1.0.0".encode()).hexdigest() for cid,date in zip(out.customer_id,out.snapshot_date)]; out["churn_probability"]=score_prob; out["predicted_churn"]=(score_prob>=threshold).astype(int); out["model_version"]="1.0.0"; out.to_csv(TABLE_DIR/"12-batch-predictions.csv",index=False,date_format="%Y-%m-%d")
    baseline=[]
    for feature in NUMERIC: baseline.append({"item":feature,"type":"numeric","mean":development[feature].mean(),"standard_deviation":development[feature].std(),"missing_rate":development[feature].isna().mean()})
    baseline.append({"item":"churn_probability","type":"prediction","mean":test_prob.mean(),"standard_deviation":test_prob.std(),"missing_rate":0}); pd.DataFrame(baseline).to_csv(TABLE_DIR/"12-monitoring-baseline.csv",index=False)
    REPORT_PATH.write_text(f"# Project Handoff: Monthly Retention Support\n\nSelected logistic regression; threshold {threshold:.2f}; final ROC AUC {metrics.iloc[0].roc_auc:.3f}. The model supports voluntary human-reviewed outreach only. It must not drive adverse automated action. Artifact SHA-256: `{digest}`. Monitor schema validity, feature and prediction drift, mature-label performance, subgroup errors, overrides, and complaints. Pause use after schema failure, material performance degradation, or unresolved subgroup harm.\n",encoding="utf-8")
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    roles=["Train"]*5+["Validate","Test","Score"]; colors={"Train":"#4C78A8","Validate":"#F28E2B","Test":"#59A14F","Score":"#9C9C9C"}; axes[0,0].bar([d.strftime("%b") for d in dates],[1]*8,color=[colors[r] for r in roles]); axes[0,0].set_title("Temporal roles"); axes[0,0].set_yticks([])
    axes[0,1].bar(comparison.model,comparison.validation_roc_auc,color=["#9C9C9C","#4C78A8","#F28E2B"]); axes[0,1].tick_params(axis="x",rotation=20); axes[0,1].set_ylim(0,1); axes[0,1].set_title("Validation ROC AUC")
    for m,c in [("precision","#59A14F"),("recall","#E15759"),("f1","#4C78A8")]:axes[0,2].plot(thresholds.threshold,thresholds[m],label=m.title(),color=c)
    axes[0,2].axvline(threshold,ls="--",color="#222"); axes[0,2].set_title("Validation threshold trade-off"); axes[0,2].legend(frameon=False)
    cm=np.array([[tn,fp],[fn,tp]]); image=axes[1,0].imshow(cm,cmap="Blues"); axes[1,0].set_xticks([0,1],["No churn","Churn"]); axes[1,0].set_yticks([0,1],["No churn","Churn"]); axes[1,0].set_title("Final-test confusion matrix")
    for i in range(2):
        for j in range(2):axes[1,0].text(j,i,str(cm[i,j]),ha="center",va="center")
    shown=importance.sort_values("mean_roc_auc_decrease"); axes[1,1].barh(shown.feature,shown.mean_roc_auc_decrease,xerr=shown.sd_roc_auc_decrease,color="#59A14F"); axes[1,1].set_title("Held-out permutation importance")
    axes[1,2].hist(score_prob,bins=20,color="#4C78A8",edgecolor="white"); axes[1,2].axvline(threshold,ls="--",color="#222"); axes[1,2].set_title("Unlabelled scoring-batch risk")
    fig.suptitle("One decision, one reproducible machine-learning lifecycle",fontsize=17,fontweight="bold"); fig.savefig(FIGURE_PATH,dpi=180,bbox_inches="tight"); plt.close(fig)
    print(f"Saved {FIGURE_PATH.relative_to(ROOT)}"); print(f"Saved {REPORT_PATH.relative_to(ROOT)}"); print(f"Final test ROC AUC: {metrics.iloc[0].roc_auc:.3f}")


if __name__=="__main__":main()
