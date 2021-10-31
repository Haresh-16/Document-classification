1. pip install requirements.txt

2. Run FlaskAppv2.py in cmd

3. Hit "http://127.0.0.1:5000/dakshHit" in PostMan with Headers (key=Content-Type , value=multipart/form-data)
   and Body(key=image , value = (Files --> Image selected from 'Test images' in current dir)

4. Output description:
	
	defaultdict with two key-value pairs is outputted

	key- Ranking in order of precedence of correctness of alloted category

	value - list with [category_name , bounding_box_coordinates, confidence_%]
		category_name = Any one of (open manhole , overflowing garbage , patchy roads , proper road , proper bins , proper manholes)
		bounding_box_coordinates, = A rectangle highlighting the problem
		confidence_% = Confidence about correctness of output

	Example output: 
	
		{
    "1": [ 
        [
            "open manhole",
            [
                0,
                140,
                496,
                219
            ],
            "99.97088313102722% Confidence"
        ]
    ],
    "2": [
        [
            "proper manhole",
            [
                209,
                138,
                280,
                183
            ],
            "0.02701951307244599% Confidence"
        ]
    ]
}