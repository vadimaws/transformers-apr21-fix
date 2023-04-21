import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

'''
import datasets
import evaluate
'''
from datasets import load_dataset
import pandas as pd
import numpy as np

'''
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions
'''
'''
import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
'''

# load the dataset.
raw_datasets = load_dataset("race", "all")
column_names = raw_datasets["train"].column_names
print('column names: ', column_names)
'''
column_names = raw_datasets["validation"].column_names
column_names = raw_datasets["test"].column_names
'''

# names of columns.
exampleID_column_name = "example_id" if "example_id" in column_names else column_names[0]
article_column_name = "article" if "article" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]
question_column_name = "question" if "question" in column_names else column_names[3]
options_column_name = "options" if "options" in column_names else column_names[4]

# train, val, test sets.
race_train = raw_datasets["train"]
race_val = raw_datasets["validation"]
race_test = raw_datasets["test"]
print(race_train.shape)
print(race_val.shape)
print(race_test.shape)

# create LISTS for each train set.
exampleID_train = list(race_train[exampleID_column_name])  # .tolist()
article_train = list(race_train[article_column_name])  # .tolist()
answer_train = list(race_train[answer_column_name])  # .tolist()
question_train = list(race_train[question_column_name])  # .tolist()
options_train = list(race_train[options_column_name])  # .tolist()

print('\n')
print(type(article_train))
print(len(article_train))
print('\n')

print(article_train[0])
print('\n')
print(article_train[1])
print('\n')
print(article_train[2])
print('\n')

print(answer_train[0])
print('\n')
print(answer_train[1])
print('\n')
print(answer_train[2])
print('\n')

print(question_train[0])
print('\n')
print(question_train[1])
print('\n')
print(question_train[2])
print('\n')

print(options_train[0])
print('\n')
print(options_train[1])
print('\n')
print(options_train[2])
print('\n')


i = 0
article0 = article_train[0]
answer0 = answer_train[0]
question0 = question_train[0]
options0 = options_train[0]

# i want a column called TEXTBOOK_TEXT.
# i want a LABELS column called MC_QUESTION.

prefix = "Based on the following passage, generate a question with one correct answer choice and three distractor choices: \n\n "
textbook_text0 = prefix + article0
print('\n')
print('TEXTBOOK TEXT: ')
print(textbook_text0)

list_ABCD = ['A. ', 'B. ', 'C. ', 'D. ']
str_mc_question = question0 + '\n'
for j in range(len(options0)):
    str_mc_question += list_ABCD[j] + options0[j] + ' '
# now append question to answer.
str_mc_question += '\nAnswer: ' + answer0
print('\n')
print('QUESTION: ')
print(str_mc_question)

text_prefix = "Based on the following passage, generate a question with one correct answer choice and three distractor choices: \n\n "
def processing_RACE_df(df, source_prefix):

    # create LISTS for each train set.
    exampleID_list = list(df[exampleID_column_name])  # .tolist()
    article_list = list(df[article_column_name])  # .tolist()
    answer_list = list(df[answer_column_name])  # .tolist()
    question_list = list(df[question_column_name])  # .tolist()
    options_list = list(df[options_column_name])  # .tolist()

    # go into a loop for each datapoint in df.
    num_dp = df.shape[0]
    list_textbook_texts = []
    list_mc_questions = []
    ABCD_list = ['A. ', 'B. ', 'C. ', 'D. ']

    # go into loop for each datapoint.
    for k in range(num_dp):
        curr_article = article_list[k]
        curr_answer = answer_list[k]
        curr_question = question_list[k]
        curr_options = options_list[k]

        # create textbook_text datapoint.
        curr_textbook_text = source_prefix + curr_article
        list_textbook_texts.append(curr_textbook_text)

        # go into loop for each answer choice.
        curr_mc_question = curr_question + '\n'
        for l in range(len(curr_options)):
            """ str_mc_question += list_ABCD[j] + options0[j] + ' ' """
            curr_mc_question += ABCD_list[l] + curr_options[l] + ' '
        curr_mc_question += '\nAnswer: ' + curr_answer
        list_mc_questions.append(curr_mc_question)

    # now, create a NEW df of 2 columns.
    df_new = pd.DataFrame(columns=['INPUT_textbook_text', 'OUTPUT_mc_question'])
    df_new['INPUT_textbook_text'] = list_textbook_texts
    df_new['OUTPUT_mc_question'] = list_mc_questions

    # return modified DF, with two new columns.
    return df_new


# let's see if this works with the race dataset.
print('\n')
print('\n')
print('\n')
print('*******************************************************')
print('BEFORE PRE-PROCESSING....')
print(type(race_train))
print(race_train)
# print(race_train.columns)
print('\n')
new_race_train = processing_RACE_df(race_train, text_prefix)
print('AFTER PRE-PROCESSING....')
print(new_race_train)
# print(new_race_train.columns)
print(new_race_train['INPUT_textbook_text'])
print(new_race_train['OUTPUT_mc_question'])


def processing_MHE_df(df):
    """
    Fill this in once you get MHE data and know how it is structured.
    """
    return 0



"""
# make sure all lengths are 4.
print('\n')
length_lists = 0
print('number of dp: ', len(options_train))
for j in range(len(options_train)):
    curr_options_list = options_train[j]
    len_list = len(curr_options_list)
    if len_list == 4:
        length_lists += 1
print('# of lists of length 4: ', length_lists)
# so all have length 4. (this could eventually be a problem, but is good now.)
"""


# figure this out.
# finish the training script.
# start fine-tuning.


# FIGURE OUT PADDING.

"""
def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=data_args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


if training_args.do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        # We will select sample from whole data if argument is specified
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    # Create train feature from dataset
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    if data_args.max_train_samples is not None:
        # Number of samples might increase during Feature Creation, We select only specified max samples
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
"""


