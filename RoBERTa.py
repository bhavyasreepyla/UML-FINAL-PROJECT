from google.colab import drive
drive.mount('/content/drive')

!pip install transformers datasets accelerate imbalanced-learn -q

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Ready! Device: {device}")

DRIVE_PATH = '/content/drive/MyDrive/UML_Project/'

df = pd.read_csv(DRIVE_PATH + 'EDA_data-FULL.csv')

valid_labels = ['update-me', 'give-me-perspective', 'educate-me', 'connect-me', 'inspire-me', 'help-me']
df_filtered = df[df['User_Needs'].isin(valid_labels)].copy().reset_index(drop=True)

# Combine title + section + text for maximum signal
def make_text(row):
    section = str(row['Section']) if pd.notna(row['Section']) else ''
    title = str(row['Title']) if pd.notna(row['Title']) else ''
    text = str(row['text']) if pd.notna(row['text']) else ''
    return f"{section} | {title} | {text}"[:1500]

df_filtered['input_text'] = df_filtered.apply(make_text, axis=1)

# Label encoding
labels = sorted(df_filtered['User_Needs'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}
df_filtered['label'] = df_filtered['User_Needs'].map(label2id)

print(f"✅ {len(df_filtered)} articles")
print(df_filtered['User_Needs'].value_counts())

train_df, test_df = train_test_split(
    df_filtered[['input_text', 'label']],
    test_size=0.2,
    random_state=42,
    stratify=df_filtered['label']
)

# Oversample minority classes in training set
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(
    train_df['input_text'].values.reshape(-1, 1),
    train_df['label'].values
)

train_balanced = pd.DataFrame({
    'input_text': X_res.flatten(),
    'label': y_res
})

print(f"✅ Train (balanced): {len(train_balanced)}")
print(f"✅ Test: {len(test_df)}")

model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)
model = model.to(device)

def tokenize(examples):
    return tokenizer(
        examples['input_text'],
        truncation=True,
        max_length=512,
        padding=False
    )

train_dataset = Dataset.from_pandas(train_balanced.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['input_text'])
test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['input_text'])

train_dataset.set_format('torch')
test_dataset.set_format('torch')

print(f"✅ RoBERTa loaded! Parameters: {model.num_parameters():,}")
print(f"✅ Train: {len(train_dataset)} | Test: {len(test_dataset)}")

from torch import nn
import numpy as np

# Calculate class weights to fix weak classes
class_counts = np.bincount(train_balanced['label'].values)
total = class_counts.sum()
class_weights = torch.tensor(
    [total / (len(class_counts) * c) for c in class_counts],
    dtype=torch.float32
).to(device)

print("Class weights:")
for i, (label, weight) in enumerate(zip(labels, class_weights)):
    print(f"  {label}: {weight:.2f}")

# Custom trainer with weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted')
    }

training_args = TrainingArguments(
    output_dir='./roberta_results',
    num_train_epochs=10,               # More epochs, no early stopping
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=300,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=3,
    logging_steps=50,
    report_to="none",
    fp16=True,
    dataloader_num_workers=2,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
    # No early stopping this time
)

print("\n🚀 Training started!")
print("10 epochs, weighted loss, no early stopping")
print("Should hit 80%+ by epoch 6-8\n")

trainer.train()

preds_output = trainer.predict(test_dataset)
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = test_df['label'].values

acc = accuracy_score(y_true, y_pred)

print(f"\n{'='*50}")
print(f"  FINAL ACCURACY: {acc*100:.1f}%")
print(f"{'='*50}\n")
print(classification_report(y_true, y_pred, target_names=labels))
