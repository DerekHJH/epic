from datasets import load_dataset
from benchmarks_ours.data_sets.data_set import Data_set
from typing import Dict, List, Any
import logging
import re
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

# Manually label where the answer comes from for the first 120 data points
ref_doc_ids = [
    [7], [3,4], [6], [2,8], [9], [], [6], [6], [8], [3,6],
    [0,5], [3], [1], [1,9], [6], [8], [2,6], [0,1], [1,3], [2], 
    [0,8], [0,1], [2,5,6], [3,6], [0,3,6,7], [3,7], [4,7], [3,7], [4,6], [0,8],
    [6,7,9], [2,3], [4], [0,4,9], [0,1,3,5], [4,7], [0,2,4], [2,3], [2,6], [6,8],
    [7], [0,2,3,7], [0], [5,7,8], [0,8], [3], [2,4,5], [], [3,8], [3,5], 
    [3,6], [6,8], [1,7], [0,5], [0,4,6], [], [0,5], [3,7], [2,3], [4,7],
    [6,8], [5,7,8], [2,4], [3,7], [0,9], [3,5,6,9], [4,8], [2,3], [3,7,8,9], [1,2],
    [0,8], [2,9], [6,9], [0,6], [3], [0,8], [2,8], [1,6], [0,2], [3,7],
    [2, 8], [4, 6], [9, 4, 2], [6 ,7], [1, 6], [0, 1, 4, 3], [6, 8], [3, 5], [2], [8, 9], 
    [0, 2, 3, 8], [5, 7], [2, 6], [4], [3, 4], [2, 5], [6, 9], [4, 5], [0, 2, 4, 7], [0, 2],
    [3,4], [6,9], [0,2], [3,4], [6,8], [1,4], [6,8], [1,5], [9], [0,1,6,9],
    [3,9], [5,8], [2,5,7], [0,1], [2,9], [1,3,9], [0,2], [5,7], [1,4,5,9], [5,7],
    [0,1], [1,5,6,9], [3,9], [0,9], [0,2,8,9], [0,2,6,9], [3,6,7,8], [2,3], [1,4], [1,5],
    [4,8], [1,9], [5,6], [4,7], [3,5,6,7], [2,6,7,9], [1,3], [1,5], [3,7,8,9], [2,3,6,7],
    [1,5], [3,8], [5,9], [0,1,3,4], [6,8], [1,5], [1,8], [6,7], [4,6], [5,6], 
    [7,9], [1,4], [7,9], [0,8], [0,2], [3,8], [1,2], [4,8], [2,3], [0,1], 
    [2,4], [2,3], [0,5,7,8], [0,5], [1,2], [1,8], [4,6], [0,3,7,8], [5,6], [2,7],
    [8,9], [1,4], [0,1,8,9], [4,8], [6,7], [1,3], [0,4,5,6], [2,8], [3,8], [0,3], 
    [0,3], [7,9], [1,3], [2,7], [6,7], [1,5], [1,3,4,5], [0,1,8,9], [2,5], [4,8], 
    [0,8], [0,1], [2,4], [6,8], [0,8], [6,8], [6,8], [4,7], [1,5], [0,2]
]
assert len(ref_doc_ids) == 200
class WikiMQA(Data_set):

    def load_data_from_hf(self):
        return load_dataset('THUDM/LongBench', '2wikimqa', split='test').to_pandas()
    
    def reorder_docs(self, ref_doc_pos: str):
        self.data['ref_doc_ids'] = ref_doc_ids
        self.data = self.data.apply(lambda row: self._reorder_docs(row, ref_doc_pos), axis=1)

    def _reorder_docs(self, row: Dict, ref_doc_pos: str) -> Dict:
        """
        Reorder the documents based on the ref_doc_pos.
        Assume row['ref_doc_ids'] is prepared
        """
        docs = row['context'].copy()
        ref_docs = [docs[i] for i in row['ref_doc_ids']]
        for i in range(len(ref_docs)):
            docs.remove(ref_docs[i])
        match ref_doc_pos:
            case 'beginning':
                docs = ref_docs + docs
            case 'middle':
                docs = docs[:len(docs)//2] + ref_docs + docs[len(docs)//2:]
            case 'end':
                docs = docs + ref_docs
            case _:
                # Treat all other cases as 'original'
                docs = row['context']
        row['context'] = docs
        return row
            
    def _split_docs(self, row: Dict) -> Dict:
        all_docs: str = row['context']
        delimiter = r'(Passage \d+:\n)'
        docs: List[str] = re.split(delimiter, all_docs)
        assert docs[0] == '', f'First element of docs is not empty: {docs[0]}'
        docs = docs[1:]
        row['context'] = row['context'] = [docs[i] + docs[i+1] for i in range(0, len(docs), 2)]
        return row

    def _append_input(self, row: Dict) -> Dict:
        row['input'] = '\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: ' + row['input'] + '\nAnswer within 5 words:'
        return row

    def _append_system_prompt(self, row: Dict) -> Dict:
        row['system_prompt'] = 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n'
        return row
    
    def _custom_process_data(self) -> None:
        if 'ref_doc_pos' in self.kwargs:
            self.reorder_docs(self.kwargs['ref_doc_pos'])

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    dataset = WikiMQA(tokenizer=tokenizer)
    import pdb; pdb.set_trace()
    

        

        
"""Manually label where the answer comes from for the first 60 data points
dataset['data_id'][0] = '41ac2a4beb0af8f58d01863a62b90692f7c7d74b5e3a58d9'
dataset['reference_doc_id'][0] = [7]
dataset['cross-doc'][0] = None

dataset['data_id'][1] = '43924e4ac5039ce3fadda49604bfcb0f5238af81774616e53'
dataset['reference_doc_id'][1] = [3,4] # Not independent!
dataset['cross-doc'][1] = [3,4]

dataset['data_id'][2] = '2c952e3e1ca394df975103b3135b3c38e0ee16e25d860258'
dataset['reference_doc_id'][2] = [6]
dataset['cross-doc'][2] = None

dataset['data_id'][3] = 'aec83da1f2faf6ec8badfd53d632f525c9ef2090d99d1c6c'
dataset['reference_doc_id'][3] = [2,8]
dataset['cross-doc'][3] = [2,8]

dataset['data_id'][4] = '4b28d517ce1c1e3cfec9282ca7b212c1cb87c254781d7c86'
dataset['reference_doc_id'][4] = [9] # Wrong Answer!
dataset['cross-doc'][4] = None  

dataset['data_id'][5] = '81cd47ec621f2228d8bb4ec351ffd6b23d23107e5b287ffb'
dataset['reference_doc_id'][5] = None # The question&answer are both strange and wrong!
dataset['cross-doc'][5] = None # I can't find the answer in the reference document!

dataset['data_id'][6] = '048e82b64b5651b74d452db7151c2110a718128dfd12a774'
dataset['reference_doc_id'][6] = [6]
dataset['cross-doc'][6] = None

dataset['data_id'][7] = 'c8fd8db4d295aef41a9434299a8eeffb9af5e2bbcde4f13a'
dataset['reference_doc_id'][7] = [6]
dataset['cross-doc'][7] = None

dataset['data_id'][8] = '4db5bcd1d49fce674507d9128850eb71b808b7dc4246e882'
dataset['reference_doc_id'][8] = [8]
dataset['cross-doc'][8] = None

dataset['data_id'][9] = 'be97290f663a83ba27007dd262ca2a6072c9156f775a24ad'
dataset['reference_doc_id'][9] = [3,6]
dataset['cross-doc'][9] = None

dataset['data_id'][10] = '538d0eac8ec082e93d6f273237fafd1f23fb405e35bad84e'
dataset['reference_doc_id'][10] = [0,5]
dataset['cross-doc'][10] = [0,5]

dataset['data_id'][11] = '37022dee1597491021c2aa7522276a600eb0dd316f12cbb6'
dataset['reference_doc_id'][11] = [3]
dataset['cross-doc'][11] = None

dataset['data_id'][12] = '90f7f51f0c27f041e730cf8893f79f3f40fcb0aab1820a4d'
dataset['reference_doc_id'][12] = [1]
dataset['cross-doc'][12] = None

dataset['data_id'][13] = 'ee29eb2556135405fa19a48b0abdff0aceddef7594169c33'
dataset['reference_doc_id'][13] = [1,9] # Wrong Answer!
dataset['cross-doc'][13] = None

dataset['data_id'][14] = '2a398d925d3607c94ffb2d0cf9fe2fe6da6e9970ce95578a'
dataset['reference_doc_id'][14] = [6]
dataset['cross-doc'][14] = None

dataset['data_id'][15] = '8d3cf7916a2f8a2762de13a9520b6350e46464fbf7f63dd5'
dataset['reference_doc_id'][15] = [8]
dataset['cross-doc'][15] = None

dataset['data_id'][16] = '5844352bd3fe5898756970abec3099ec694be6ad15c498db'
dataset['reference_doc_id'][16] = [2,6]
dataset['cross-doc'][16] = [2,6]

dataset['data_id'][17] = '4413705182b57f12dc553e0919b50fd3c3b111cb4d82d12f'
dataset['reference_doc_id'][17] = [0,1]
dataset['cross-doc'][17] = [0,1]

dataset['data_id'][18] = 'cc32d31bbeb3e0d8787e963cf0843ae6b22f33817eb5a587'
dataset['reference_doc_id'][18] = [1,3]
dataset['cross-doc'][18] = [1,3]

dataset['data_id'][19] = 'b3845fcad97309850a76e3720cd829a45d2bb12ee29a9cfb'
dataset['reference_doc_id'][19] = [2]
dataset['cross-doc'][19] = None


# x--y means we require x and y to get a relatively meaningful info, and the doc id starts from 1 instead of 0
dataset['reference_doc_id'][20] = [1, 9] # 1--9
dataset['reference_doc_id'][21] = [1, 2] # 1--2
dataset['reference_doc_id'][22] = [3, 6, 7] # 3--7
dataset['reference_doc_id'][23] = [4, 7]
dataset['reference_doc_id'][24] = [1, 4, 7, 8] # 1--4; 7--8
dataset['reference_doc_id'][25] = [4, 8]
dataset['reference_doc_id'][26] = [5, 8]
dataset['reference_doc_id'][27] = [4, 8]
dataset['reference_doc_id'][28] = [4, 7] # 2--7
dataset['reference_doc_id'][29] = [1, 9] # 1--9
dataset['reference_doc_id'][30] = [7, 8, 10]
dataset['reference_doc_id'][31] = [3, 4] # 3--4
dataset['reference_doc_id'][32] = [5]
dataset['reference_doc_id'][33] = [1, 5, 10] # 5--10
dataset['reference_doc_id'][34] = [1, 2, 4, 6] # 1--4; 2--6
dataset['reference_doc_id'][35] = [5, 8]
dataset['reference_doc_id'][36] = [1, 3, 5] # 1--5
dataset['reference_doc_id'][37] = [3, 4] # 3--4
dataset['reference_doc_id'][38] = [3, 7]
dataset['reference_doc_id'][39] = [7, 9]

                                                     _id reference_doc_id     cross-doc
40 55acb15a501d22031778ff4ccb655e787668b617d911daa9              [7]          None
41 57543e0892f90733bf97dc270cdb36eb9bf8e2230940bcb9     [0, 2, 3, 7]  [0, 2, 3, 7]
42 2d1951b62f317038d28f6ac6a2b21e9d2e7f921ea528948b              [0]          None
43 b126cad8f81aed3c5cc29c20c25b1adc7d1f33e1c4e264c3        [5, 7, 8]     [5, 7, 8]
44 188123e1b35a7de33fc44a40b55ab9ff0b59d29dc30f9148           [0, 8]        [0, 8]
45 5ff6349ea0b2ecb882f535ff3cec2f9c012d3468fb7b912e              [3]          None
46 314e76591d11d30d2bd59b2f12b671e873b786656f19956d        [2, 4, 5]     [2, 4, 5]
47 7140432f1c424674d04ea819fa63f85aa4f154d72ab66285     wrong_answer  wrong_answer
48 3292d2bcbbf8fc4816d168bcb7f81c13341198da1e88b903           [3, 8]        [3, 8]
49 5421c1e591296d64636a7e99fbfe98dddfed0f83d2053380           [3, 5]        [3, 5]
50 f67db35cb225c73bb753006bd14fab03942bbe0157a74979           [3, 6]        [3, 6]
51 96500ef697df70106798988a9065594622cdeff67156cd20           [6, 8]        [6, 8]
52 5a867841d94c9f88f3bc0c333e93484486b09341db474bd6           [1, 7]        [1, 7]
53 059802de84ebd63ee99284c9cb2b7db2d45ca66c55e68ee1           [0, 5]        [0, 5]
54 fcc4a49fbe72f1b8ffef144e2e9d85bfc8897ba2864fd521        [0, 4, 6]     [0, 4, 6]
55 633204c891118a6f3d1ae62d7b444c2ee02ad0cbae4876cc     wrong_answer  wrong_answer
56 977f59ef09f7cc4ad6c71fc18d8b60e818e07a43cd7bee92           [0, 5]        [0, 5]
57 5c3829f10f9daf565e9b5b52ca19f5c044550e916479da74           [3, 7]        [3, 7]
58 0a64d8873482d91efc595a508218c6ce881c13c95028039e           [2, 3]        [2, 3]
59 44b7f326c7b3c430968c237ee4017425d03eef586d5d6cb7           [4, 7]        [4, 7]



dataset['data_id'][60] = '85d2e8c38cc3b6f464eae6d84ef2df5f41807c9f8bf3aac8'
dataset['reference_doc_id'][60] = [6,8]
dataset['cross-doc'][60] = [6,8]

dataset['data_id'][61] = '1038dcf795ffefdb5d947e54a7e2d3b04d73895e482fd324'
dataset['reference_doc_id'][61] = [5,7,8]
dataset['cross-doc'][61] = [5,7,8]

dataset['data_id'][62] = '62616c28af945825f87a95c47c1783dc284b06ec2d447402'
dataset['reference_doc_id'][62] = [2,4]
dataset['cross-doc'][62] = [2,4]

dataset['data_id'][63] = 'a3c88e70534d79b432d32c0b53d0d181c957b353f658dc95'
dataset['reference_doc_id'][63] = [3,7]
dataset['cross-doc'][63] = [3,7]

dataset['data_id'][64] = '63df7238a0005c6bffc122ca97570a95eb1d3711abaee205'
dataset['reference_doc_id'][64] = [0,9]
dataset['cross-doc'][64] = [0,9]

dataset['data_id'][65] = 'c94e0220a7c5dadee3094df576df766b6d1a15b6c6e21011'
dataset['reference_doc_id'][65] = [3,5,6,9]
dataset['cross-doc'][65] = [3,5,6,9]

dataset['data_id'][66] = 'cab5465c3f7663e5d72c15a32bc04c8a917dac3c03828814'
dataset['reference_doc_id'][66] = [4,8]
dataset['cross-doc'][66] = [4,8]

dataset['data_id'][67] = 'be0651d84c04af5de43e28cf89e78d7e87490e637167632e'
dataset['reference_doc_id'][67] = [2,3]
dataset['cross-doc'][67] = [2,3]

dataset['data_id'][68] = 'ef73e58876b94a997d928ee13b7ac7cdb8b8aa0bac856a92'
dataset['reference_doc_id'][68] = [3,7,8,9]
dataset['cross-doc'][68] = [3,7,8,9]

dataset['data_id'][69] = '7940c60a5ff2d81b62d118253577d1d891057ca45695e91a'
dataset['reference_doc_id'][69] = [1,2]
dataset['cross-doc'][69] = [1,2]

dataset['data_id'][70] = 'a88430cef36a0222c3c30780328ff266b16325a7ec723a97'
dataset['reference_doc_id'][70] = [0,8]
dataset['cross-doc'][70] = [0,8]

dataset['data_id'][71] = '0bcff697444354703dd1e8987a709b8ed2f44bf9d6b2d320'
dataset['reference_doc_id'][71] = [2,9]
dataset['cross-doc'][71] = [2,9]

dataset['data_id'][72] = '24e27e943be014d4549674d581cdc20fdae92fa3d9cd256c'
dataset['reference_doc_id'][72] = [6,9]
dataset['cross-doc'][72] = [6,9]

dataset['data_id'][73] = '0c43156b7d6bc17425a89dc6c6d48fc8c2c72ac9eba55a87'
dataset['reference_doc_id'][73] = [0,6]
dataset['cross-doc'][73] = [0,6]

dataset['data_id'][74] = '64d6357ab735a8542112f87893a6e7c89b1b307bdb0e24c7'
dataset['reference_doc_id'][74] = [3]
dataset['cross-doc'][74] = None

dataset['data_id'][75] = '9ef1c9e836fb3844fe16369a346088af89b8da29e24aab1f'
dataset['reference_doc_id'][75] = [0,8]
dataset['cross-doc'][75] = [0,8]

dataset['data_id'][76] = 'f11ac580e22d7d0e3da7cf1c725c8def8aef4aa891baa07a'
dataset['reference_doc_id'][76] = [2,8]
dataset['cross-doc'][76] = [2,8]

dataset['data_id'][77] = '884614fa7d0fe723587d2f2677d3f2143cd13ab74391bea6'
dataset['reference_doc_id'][77] = [1,6]
dataset['cross-doc'][77] = [1,6]

dataset['data_id'][78] = 'ec4ba162decfc8dd03ccbca7832f3cecf985a5542e70dec1'
dataset['reference_doc_id'][78] = [0,2]
dataset['cross-doc'][78] = [0,2]

dataset['data_id'][79] = '9c5350cca5afdd1a7254dbc26b3642401551473f4b071ce6'
dataset['reference_doc_id'][79] = [3,7]
dataset['cross-doc'][79] = [3,7]



# results
# x -> y means continuous reasoning
# (part A) // (part B) means paralleled reasoning routines
dataset['reference_doc_id'][80] = [2, 8] # 2 -> 8
dataset['reference_doc_id'][81] = [4, 6] # 4 // 6
dataset['reference_doc_id'][82] = [9, 4, 2] # (9 -> 4) // 2
dataset['reference_doc_id'][83] = [6 ,7] # 6 // 7
dataset['reference_doc_id'][84] = [1, 6] # 1 // 6
dataset['reference_doc_id'][85] = [0, 1, 4, 3] # (1 -> 4) // (0 -> 3)
dataset['reference_doc_id'][86] = [6, 8] # 6 -> 8
dataset['reference_doc_id'][87] = [3, 5] # 3 -> 5
dataset['reference_doc_id'][88] = [2]
dataset['reference_doc_id'][89] = [8, 9] # 8 // 9
dataset['reference_doc_id'][90] = [0, 2, 3, 8] # 0 // 2 // 8 // 3
dataset['reference_doc_id'][91] = [5, 7] # 5 // 7
dataset['reference_doc_id'][92] = [2, 6] # 2 -> 6
dataset['reference_doc_id'][93] = [4] # Confusing Question!
dataset['reference_doc_id'][94] = [3, 4] # 4 -> 3
dataset['reference_doc_id'][95] = [2, 5] # 2 -> 5
dataset['reference_doc_id'][96] = [6, 9] # 9 // 6
dataset['reference_doc_id'][97] = [4, 5] # 5 -> 4
dataset['reference_doc_id'][98] = [0, 2, 4, 7] # (0 -> 7) // (4 -> 2)
dataset['reference_doc_id'][99] = [0, 2] # 2 -> 0


                                                     _id reference_doc_id     cross-doc
100  259c17d6e4a5e4c74669836fec3c6de0730bf65e67e6d988           [3, 4]        [3, 4]
101  8af3f07b73fabc0cc4ef4e6928818d1ff08ead9f41a5085a           [6, 9]        [6, 9]
102  3816f7d8e90a5497a1c0e415bf25218037c690e8e4d0d016           [0, 2]        [0, 2]
103  025911ff0afd34976b0a800c198c4117f95edfcd7b64bb24           [3, 4]        [3, 4]
104  cdf0b77af7d55cc9a3424613604e8dbabe7a01e4869ad502           [6, 8]        [6, 8]
105  1ff4dc6c1f7c02445945ced1b35a08544d7ba76089675883           [1, 4]        [1, 4]
106  c7347f362602f3a0be4b051c0b68219e0648afa64772bd30           [6, 8]        [6, 8]
107  0cf27c68148c99571860af561ce5f64c4524fde278defc63           [1, 5]        [1, 5]
108  93ff7c9369b6ff2f37896b54f6e4d8ff91c125f78ae8bf9a              [9]          None
109  550f734c7ef908fcfe6b1b7e761826340f00abaa5eb305a0     [0, 1, 6, 9]  [0, 1, 6, 9]
110  860a0ad0039969cc7f01d512df2b0ea283d8f7d0c08886e5           [3, 9]        [3, 9]
111  3723ec96e3b1af88c73edaa0a5e87f20194d9d212a67b615           [5, 8]        [5, 8]
112  60735709da9cbdd65bab0a11d1a0991ec842fe74d1ab583b        [2, 5, 7]     [2, 5, 7]
113  76a7a3db55c966560ad3e8d34037602d4d12714980898b78           [0, 1]        [0, 1]
114  efea9c05de081efb7ef76741d3f24da2e993d54d99677b72           [2, 9]        [2, 9]
115  435fae9b23d8b22e85a57cfd8ceb8541dea5b0c551acc3a9        [1, 3, 9]     [1, 3, 9]
116  3ee59c959e64a29c8a30263816ad47b76ffab36e5b8859ee           [0, 2]        [0, 2]
117  43cf26663f3b4cec7a8ea68d56909201a2d1b1ad82512190           [5, 7]        [5, 7]
118  77a79ad1902a4732e8030d79380395b7e614d65de3cdd038     [1, 4, 5, 9]  [1, 4, 5, 9]
119  f0523a41000b43996413780968b611c3f4d28b87f4acc859           [5, 7]        [5, 7]


dataset['data_id'][120] = '6a60fa19187345478a447b63d8f52a1cb5341e964cda29db'
dataset['reference_doc_id'][120] = [0,1]
dataset['cross-doc'][120] = [0,1]

dataset['data_id'][121] = '78227cb74e7b2401070e50abff8069510f5e5ccec0a2479f'
dataset['reference_doc_id'][121] = [1,5,6,9]
dataset['cross-doc'][121] = [1,5,6,9]

dataset['data_id'][122] = 'f7abcc33c22de53769935182a278a199e9f263a12e56673e'
dataset['reference_doc_id'][122] = [3,9] # wrong answer!
dataset['cross-doc'][122] = [3,9]

dataset['data_id'][123] = '24a43c1b071d8bb99f51fec368dbc438cf7aea1f549ea195'
dataset['reference_doc_id'][123] = [0,9]
dataset['cross-doc'][123] = [0,9]

dataset['data_id'][124] = '89530e21fb2f36eb499cea025efa9c80b10049f72945bfb6'
dataset['reference_doc_id'][124] = [0,2,8,9]
dataset['cross-doc'][124] = [0,2,8,9]

dataset['data_id'][125] = 'e8c2e5c8d393b186e21c75788f7e93efeaef12a9c44880d9'
dataset['reference_doc_id'][125] = [0,2,6,9]
dataset['cross-doc'][125] = [0,2,6,9]

dataset['data_id'][126] = 'd7f6203bfe33e28a517ed4214f515fb80671c48edd4720a6'
dataset['reference_doc_id'][126] = [3,6,7,8]
dataset['cross-doc'][126] = [3,6,7,8]

dataset['data_id'][127] = '517da14d56782303ccf8ec057b86bf43169b6f2edcd6d0f1'
dataset['reference_doc_id'][127] = [2,3]
dataset['cross-doc'][127] = [2,3]

dataset['data_id'][128] = '300ccba91f776e583d418d6d9ecc5e1bdca7bd80580cd36e'
dataset['reference_doc_id'][128] = [1,4]
dataset['cross-doc'][128] = [1,4]

dataset['data_id'][129] = '81c8cd41355e5f0489dad4010b5fd414b817f7a9134affc2'
dataset['reference_doc_id'][129] = [1,5]
dataset['cross-doc'][129] = [1,5]

dataset['data_id'][130] = '0c8d9e57340d06b13e75f1dd42ddb5f08c1203fb62632353'
dataset['reference_doc_id'][130] = [4,8]
dataset['cross-doc'][130] = [4,8]

dataset['data_id'][131] = '544828db02efd1313bfc0a1bdeb8c3df5985c299254b0222'
dataset['reference_doc_id'][131] = [1,9]
dataset['cross-doc'][131] = [1,9]

dataset['data_id'][132] = 'f5cf6d94b9e4e7adfc30d262396bc29e201d9e07ead1d088'
dataset['reference_doc_id'][132] = [5,6]
dataset['cross-doc'][132] = [5,6]

dataset['data_id'][133] = '018c9c2d781ea2c9d57a31bd2e0ff30b55e77017a319484a'
dataset['reference_doc_id'][133] = [4,7]
dataset['cross-doc'][133] = [4,7]

dataset['data_id'][134] = '93ef40c0bfa088c2a2dd3302976a950cba3e8d97767f65ce'
dataset['reference_doc_id'][134] = [3,5,6,7] # wrong answer!
dataset['cross-doc'][134] = [3,5,6,7]

dataset['data_id'][135] = 'e15605da0c5b6f02c85aff9f1f683d3f85d211c083792ecb'
dataset['reference_doc_id'][135] = [2,6,7,9]
dataset['cross-doc'][135] = [2,6,7,9]

dataset['data_id'][136] = 'bf2899717e8d7d3d83d865ca5ff59e26d7f647e6025e2d52'
dataset['reference_doc_id'][136] = [1,3]
dataset['cross-doc'][136] = [1,3]

dataset['data_id'][137] = '57db7adad0b62b8a396fd45937030d876ecd02153b8a931e'
dataset['reference_doc_id'][137] = [1,5]
dataset['cross-doc'][137] = [1,5]

dataset['data_id'][138] = 'ce6a3aae400b3f5f028504fec37d7be8b447bd0cb1a0cff5'
dataset['reference_doc_id'][138] = [3,7,8,9]
dataset['cross-doc'][138] = [3,7,8,9]

dataset['data_id'][139] = 'c997d53083dff6918a3f066cc76c824e43eb4ea21de33c51'
dataset['reference_doc_id'][139] = [2,3,6,7]
dataset['cross-doc'][139] = [2,3,6,7]

dataset['data_id'][140] = '184929a3ff00c41a966da9d53444da794af70ce90d624205'
dataset['reference_doc_id'][140] = [1,5]
dataset['cross-doc'][140] = [1,5]

dataset['data_id'][141] = '56db201f810a3d01b7d14e8e9e6dd94820ecaf2f4a86b2b9'
dataset['reference_doc_id'][141] = [3,8]
dataset['cross-doc'][141] = [3,8]

dataset['data_id'][142] = 'fefa6cc344f63fdceda9662af921fdcdf645abb8fe87ea73'
dataset['reference_doc_id'][142] = [5,9]
dataset['cross-doc'][142] = [5,9]

dataset['data_id'][143] = '9b0dc01a1388db020e5c5c428a002d099f8a35544e9b6261'
dataset['reference_doc_id'][143] = [0,1,3,4]
dataset['cross-doc'][143] = [0,1,3,4]

dataset['data_id'][144] = '34896de8d6a2c70e09b6dc3d113475a58899961d2aa356b3'
dataset['reference_doc_id'][144] = [6,8]
dataset['cross-doc'][144] = [6,8]

dataset['data_id'][145] = '0f629fbc7a6c3689ee9771af3aa81400674e673f11d47f2d'
dataset['reference_doc_id'][145] = [1,5]
dataset['cross-doc'][145] = [1,5]

dataset['data_id'][146] = '13dde588b81d1de4a9984418bf3d6791de7d1b1edf21d61b'
dataset['reference_doc_id'][146] = [1,8]
dataset['cross-doc'][146] = [1,8]

dataset['data_id'][147] = 'cf56ec8180c3c7622a439e68704fc1ff73770f2e8ee2926e'
dataset['reference_doc_id'][147] = [6,7]
dataset['cross-doc'][147] = [6,7]

dataset['data_id'][148] = '78d3bf98f7b05622b428940d273b8c493f24a4468d1998dd'
dataset['reference_doc_id'][148] = [4,6]
dataset['cross-doc'][148] = [4,6]

dataset['data_id'][149] = '9b0b29e391c61c570b20fae813486bc757c1fb7d25d83049'
dataset['reference_doc_id'][149] = [5,6]
dataset['cross-doc'][149] = [5,6]

dataset['data_id'][150] = 'bdb42f1c30e16bbcdf6f0ab7c1f76765ddaa5817ee5e674b'
dataset['reference_doc_id'][150] = [7,9]
dataset['cross-doc'][150] = [7,9]

dataset['data_id'][151] = 'a7b5c5d84d974a849b0d042c435bbb3fe90ff50fd5d73f79'
dataset['reference_doc_id'][151] = [1,4]
dataset['cross-doc'][151] = [1,4]

dataset['data_id'][152] = 'd0b11838d26195f89d1568431f88a02b4f4d8c9cdf99e04a'
dataset['reference_doc_id'][152] = [7,9]
dataset['cross-doc'][152] = [7,9]

dataset['data_id'][153] = 'ca72ce55dc8f774617c794681dd7062eadfd14e679418cfa'
dataset['reference_doc_id'][153] = [0,8]
dataset['cross-doc'][153] = [0,8]

dataset['data_id'][154] = '10dc2bd4f7fb1b834c7f563ae3d7af5d269d84b3dfc0b056'
dataset['reference_doc_id'][154] = [0,2]
dataset['cross-doc'][154] = [0,2]

dataset['data_id'][155] = 'bbaa4eb6b32bad24fc2b8639d03be8c16d1140ec8454b16a'
dataset['reference_doc_id'][155] = [3,8]
dataset['cross-doc'][155] = [3,8]

dataset['data_id'][156] = 'bc596d2ebcd25e79ca9cd689e50746e42e9a6fc579afc683'
dataset['reference_doc_id'][156] = [1,2]
dataset['cross-doc'][156] = [1,2]

dataset['data_id'][157] = 'f73db1873c311b2a38f27a52c5e05b021da0c2fe9590dac3'
dataset['reference_doc_id'][157] = [4,8]
dataset['cross-doc'][157] = [4,8]

dataset['data_id'][158] = '53f6acc70a61443a93c25c91ca7dc7f8780046d02e28c94a'
dataset['reference_doc_id'][158] = [2,3]
dataset['cross-doc'][158] = [2,3]

dataset['data_id'][159] = '88e4806d6c6553000e3c1ff8be5c90f5309342954ce4acae'
dataset['reference_doc_id'][159] = [0,1]
dataset['cross-doc'][159] = [0,1]

dataset['data_id'][160] = '25b5c6893c13f6e52ddd541c03f5bd6bb7b17cfeb15d9f16'
dataset['reference_doc_id'][160] = [2,4]
dataset['cross-doc'][160] = [2,4]

dataset['data_id'][161] = '3cff7074281c58717aa65021d875cc1e6a6321f71cd77742'
dataset['reference_doc_id'][161] = [2,3]
dataset['cross-doc'][161] = [2,3]

dataset['data_id'][162] = '13c4f5388b670f6c20b168e782f7478616fa3ba2edca3081'
dataset['reference_doc_id'][162] = [0,5,7,8]
dataset['cross-doc'][162] = [0,5,7,8]

dataset['data_id'][163] = '0c35123885c26fc474a7cd8e7c03fe75bfff4fc15845d3b9'
dataset['reference_doc_id'][163] = [0,5]
dataset['cross-doc'][163] = [0,5]

dataset['data_id'][164] = '2a0554b724822b6c611539e39b44381f752acdecce4868ad'
dataset['reference_doc_id'][164] = [1,2]
dataset['cross-doc'][164] = [1,2]

dataset['data_id'][165] = 'acb93469117282f2d0131e19f6210e007d681bc7a78e7b64'
dataset['reference_doc_id'][165] = [1,8]
dataset['cross-doc'][165] = [1,8]

dataset['data_id'][166] = 'a7d0cfa55dce1e96490d1e6c0188398864724e96a44e4b79'
dataset['reference_doc_id'][166] = [4,6]
dataset['cross-doc'][166] = [4,6]

dataset['data_id'][167] = '7a35ba110f4c26f7eb9d4dcc2e8c1baac383c777f0904976'
dataset['reference_doc_id'][167] = [0,3,7,8]
dataset['cross-doc'][167] = [0,3,7,8]

dataset['data_id'][168] = 'b9a9b860c0f0430853d1db2756095da274d522beed2acbeb'
dataset['reference_doc_id'][168] = [5,6]
dataset['cross-doc'][168] = [5,6]

dataset['data_id'][169] = 'a49e29926d7019e6cbaf6b0610453eda22fa1ce9043f5829'
dataset['reference_doc_id'][169] = [2,7]
dataset['cross-doc'][169] = [2,7]

dataset['data_id'][170] = '29ec971fbdf0df2d0c41c497de945024db6159c0c30be307'
dataset['reference_doc_id'][170] = [8,9]
dataset['cross-doc'][170] = [8,9]

dataset['data_id'][171] = 'de14d0bf7b2c73f7a21875677eed84a1ef270d2f29dc5b99'
dataset['reference_doc_id'][171] = [1,4]
dataset['cross-doc'][171] = [1,4]

dataset['data_id'][172] = '49035ecce6e78a355307dd6a1ddb11a87cab76090d655d2f'
dataset['reference_doc_id'][172] = [0,1,8,9]
dataset['cross-doc'][172] = [0,1,8,9]

dataset['data_id'][173] = 'd04cc75917c9ef6b7a78b82491229023013ef3cea72bfba4'
dataset['reference_doc_id'][173] = [4,8]
dataset['cross-doc'][173] = [4,8]

dataset['data_id'][174] = '8848d4bd1425f8bef31b6c279ec378bcbca14f10b702c4ea'
dataset['reference_doc_id'][174] = [6,7]
dataset['cross-doc'][174] = [6,7]

dataset['data_id'][175] = '72132e87102859b583fb69a8a2a6a8b026be4fec5a1437e0'
dataset['reference_doc_id'][175] = [1,3]
dataset['cross-doc'][175] = [1,3]

dataset['data_id'][176] = '0c15fb826ac59bbde9df698db9c277def2bc72fa5b7b497d'
dataset['reference_doc_id'][176] = [0,4,5,6]
dataset['cross-doc'][176] = [0,4,5,6]

dataset['data_id'][177] = 'cf5321576ab8cc1b8453991abac66aa5d46155d8ffa7ee17'
dataset['reference_doc_id'][177] = [2,8]
dataset['cross-doc'][177] = [2,8]

dataset['data_id'][178] = '6110cb2d4156df8d180fdd01e5f340c4731f7d1f4583fc5a'
dataset['reference_doc_id'][178] = [3,8]
dataset['cross-doc'][178] = [3,8]

dataset['data_id'][179] = 'b1034d8bbeeba5f9001f48311c182d2a20782462f0c2ed65'
dataset['reference_doc_id'][179] = [0,3]
dataset['cross-doc'][179] = [0,3]

dataset['data_id'][180] = '7218faa7d718853f8df4329a7242806cf70e63ece56f8cdb'
dataset['reference_doc_id'][180] = [0,3]
dataset['cross-doc'][180] = [0,3]

dataset['data_id'][181] = '78436f55ae9c254da5a54558ddd196c94850ba7a2b6fa637'
dataset['reference_doc_id'][181] = [7,9]
dataset['cross-doc'][181] = [7,9]

dataset['data_id'][182] = 'ddd046530f4979511397f82fe5391d0a646c0aa18225a98e'
dataset['reference_doc_id'][182] = [1,3]
dataset['cross-doc'][182] = [1,3]

dataset['data_id'][183] = '2ce836524b90befee94e944e77ce3420878e75b5a1e14c26'
dataset['reference_doc_id'][183] = [2,7]
dataset['cross-doc'][183] = [2,7]

dataset['data_id'][184] = 'bcd3d700ef14d9a1f6b766c5d185b43ac8f3b7feffd4732f'
dataset['reference_doc_id'][184] = [6,7]
dataset['cross-doc'][184] = [6,7]

dataset['data_id'][185] = '24591a37ff748fb849a62e6a4f4095f7089d7ced43595d01'
dataset['reference_doc_id'][185] = [1,5]
dataset['cross-doc'][185] = [1,5]

dataset['data_id'][186] = '2e86241a9d27925aa52e039067fa83e27e305301188daef7'
dataset['reference_doc_id'][186] = [1,3,4,5]
dataset['cross-doc'][186] = [1,3,4,5]

dataset['data_id'][187] = 'c1d354f1d1855b21eb3c3cd204d5030080ef9e3c7e4b0195'
dataset['reference_doc_id'][187] = [0,1,8,9]
dataset['cross-doc'][187] = [0,1,8,9]

dataset['data_id'][188] = '2c605656720054b6c9e6e0f15f99c1b4675db67e6e098cee'
dataset['reference_doc_id'][188] = [2,5]
dataset['cross-doc'][188] = [2,5]

dataset['data_id'][189] = '1edc61803f35344a5a4b6d273e6eb0f643fefcec0fa0a22a'
dataset['reference_doc_id'][189] = [4,8]
dataset['cross-doc'][189] = [4,8]


dataset['data_id'][190] = 'd65eba4a41497416d108c17f366636efb4cbda4c215c8818'
dataset['reference_doc_id'][190] = [0,8]
dataset['cross-doc'][190] = [0,8]

dataset['data_id'][191] = '2df722c08f52cbf987c84019f7f2e5455cd29992738441cc'
dataset['reference_doc_id'][191] = [0,1]
dataset['cross-doc'][191] = [0,1]

dataset['data_id'][192] = '814dddaa4a997380fc80cc1c72d48cb20545411924a0603f'
dataset['reference_doc_id'][192] = [2,4]
dataset['cross-doc'][192] = [2,4]

dataset['data_id'][193] = '1d554e7ae21ed7b5d142b57bcdab110cf6cea6f5256a58ee'
dataset['reference_doc_id'][193] = [6,8]
dataset['cross-doc'][193] = [6,8]

dataset['data_id'][194] = 'd0ebddd00bfc9c5747021ae27c0ea5cdf21355bc86b83753'
dataset['reference_doc_id'][194] = [0,8]
dataset['cross-doc'][194] = [0,8]

dataset['data_id'][195] = '8f6c8de1c1518b5bdcbc4ff1cdfe035b9207971a6362574f'
dataset['reference_doc_id'][195] = [6,8]
dataset['cross-doc'][195] = [6,8]

dataset['data_id'][196] = '59e42b2cab6da2436fae05e25c49f6f4b4893b07bb48e3c1'
dataset['reference_doc_id'][196] = [6,8]
dataset['cross-doc'][196] = [6,8]

dataset['data_id'][197] = '35cbcb102aa5b4f9188011190e0f30456df5673fcddacf49'
dataset['reference_doc_id'][197] = [4,7]
dataset['cross-doc'][197] = [4,7]

dataset['data_id'][198] = '44864cf7bbca78294d1b24c5acfa035f30849d802356aad4'
dataset['reference_doc_id'][198] = [1,5]
dataset['cross-doc'][198] = [1,5]

dataset['data_id'][199] = '67705e148a6e0540819f466bb4fdc73ed2e5cbcd64d80930'
dataset['reference_doc_id'][199] = [0,2]
dataset['cross-doc'][199] = [0,2]

"""