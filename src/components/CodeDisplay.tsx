import { Box, Flex, Show, Stack, Text } from '@chakra-ui/react';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import Code from '../entities/code';

function CodeDisplay({ codes }: { codes: Code[] }) {
  return (
    <Box pt={5} pb={2}>
      <Box 
        bg='gray.50' 
        minWidth='0' 
        py={2} 
        borderLeft='1px solid red'
        borderTop='5px solid pink'
      >
        {codes.map((code: Code) => <Stack key={code.id} p={3}>
          <Flex>
            <Show above='md'>
              <Text
                pt='6.5px'
                pr={2}
                color='gray.500'
                fontSize='13px'
                fontFamily='Menlo, monospace'
              >
                In [<span style={{ color: 'tomato' }}>{code.id}</span>]:
              </Text>
            </Show>
            <Box flex='1' overflowX='auto'>
              <SyntaxHighlighter
                language='python'
                style={docco}
                customStyle={{
                  backgroundColor: '#E2E8F0',
                  fontFamily: 'Menlo, monospace',
                  fontSize: '13px',
                  borderWidth: '1px',
                  borderColor: 'gray',
                }}>
                {code.input}
              </SyntaxHighlighter>
            </Box>
          </Flex>
          <Flex>
            <Show above='md'>
              <Text
                pt='6.5px'
                pr={4}
                color='gray.500'
                fontSize='13px'
                fontFamily='Menlo, monospace'
              >
                Out []:
              </Text>
            </Show>
            <Box flex='1' overflowX='auto'>
              <SyntaxHighlighter
                language='python'
                style={docco}
                customStyle={{
                  fontFamily: 'Menlo, monospace', 
                  fontSize: '13px',
                  backgroundColor: code.output.includes('Error')
                      ? 'pink'
                      : code.output.includes('Warning')
                      ? '#ADD8E6'
                      : '',
                }}
              >
                {code.output}
              </SyntaxHighlighter>
            </Box>
          </Flex>
        </Stack>)}
      </Box>
    </Box>
  );
}

export default CodeDisplay;

// minWidth="0": 
// to ensure it doesn’t force the content to expand beyond the screen width.