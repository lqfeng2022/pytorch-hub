import { Box, Divider, Flex, Stack, Text } from '@chakra-ui/react';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';

interface Code {
  id: number,
  input: string, 
  output: string,
}

function CodeDisplay({ codes }: { codes: Code[] }) {
  return (
    <>
      <Flex align='center' justifyContent='center' h='50px'>
        <Divider variant='middle'/>
      </Flex>
      <Box bg='gray.50'>
        {codes.map((code: Code) => <Stack py={3}>
          <Flex>
            <Text pt='6.5px' pr={2} color='gray.500' fontSize='13px' fontFamily='Menlo, monospace'>
              In [<Text color='tomato' display='inline'>{code.id}</Text>]:
            </Text>
            <Box flex='1'>
              <SyntaxHighlighter
                language='python'
                style={docco}
                customStyle={{
                  backgroundColor: '#E2E8F0',
                  fontFamily: 'Menlo, monospace',
                  fontSize: '13px',
                  borderWidth: '1px',
                  borderColor: 'gray'
                }}>{code.input}</SyntaxHighlighter>
            </Box>
          </Flex>
          <Flex>
            <Text pt='6.5px' pr={2} color='gray.500' fontSize='13px' fontFamily='Menlo, monospace'>
              Out []:
            </Text>
            <Box flex='1'>
              <SyntaxHighlighter
                language='python'
                style={docco}
                customStyle={{
                  fontFamily: 'Menlo, monospace',
                  fontSize: '13px',
                }}>{code.output}</SyntaxHighlighter>
            </Box>
          </Flex>
        </Stack>)}
      </Box>
      <Flex align='center' justifyContent='center' h='50px'>
        <Divider variant='middle'/>
      </Flex>
    </>
  );
}

export default CodeDisplay;