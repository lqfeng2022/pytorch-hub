import { Center, Box, Stack, Text, HStack } from '@chakra-ui/react'
import Section from '../entities/section'

interface Props {
  title: string,
  definition: Section
}
const Definition = ({ title, definition }: Props) => {
  return (
    <Box pt={5}>
      <Text as='b' fontSize='xl' color='gray.600'>{title}</Text>
      <Center my={5} minH='250px' bg='red.50'>
        <Box maxW='600px' px={5}>
          <Text textAlign='center' fontSize={{base: 'xl', lg: '2xl'}} color='tomato'>
            {definition.value}
          </Text>
        </Box>
      </Center>
      <Stack spacing={4}>
        {definition.content.map((p) => 
          <div key={p.id}>
            {p.title && 
              <>
                <HStack>
                  <Text>{'•'}</Text>
                  <Text as='b'>{p.title}</Text>
                </HStack>
                <Text pl={4}>{p.value}</Text>
             </>
            }
            {!p.title && <Text>{p.value}</Text>}
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default Definition