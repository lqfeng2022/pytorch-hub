import { Text } from '@chakra-ui/react';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';

interface Props {
  count: number,
  input: string, 
  output: string,
}

function CodeDisplay({ count, input, output }: Props) {
  return (
    <>
      <Text py={2} fontFamily='Menlo, monospace'>
        In [<Text color='tomato' display='inline'>{count}</Text>]:
      </Text>
      <SyntaxHighlighter 
        language='python' 
        style={docco} 
        customStyle={{ 
          backgroundColor: '#E2E8F0', 
          fontFamily: 'Menlo, monospace',
          fontSize: '15px'
        }}
      >
        {input}
      </SyntaxHighlighter>
      <Text py={2} fontFamily='Menlo, monospace'>Out []:</Text>
      <SyntaxHighlighter language='python' style={docco}>
        {output}
      </SyntaxHighlighter>
    </>
  );
}

export default CodeDisplay;