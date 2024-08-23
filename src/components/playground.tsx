import { Box, Flex, Image, SimpleGrid, Text } from '@chakra-ui/react';
import turningTest from '../assets/TuringTest.png'


const HistroyAI = () => {
  return (
    <>
      <Text as='b' fontSize='lg'>-- The History of AI?</Text>
      <SimpleGrid columns={[1, null, 2]} spacing='30px'>
        <Flex alignItems="center" justifyContent="center">
          <Image src={turningTest} />
        </Flex>
        <Box>
          <Text as='b'>1) Turning Test</Text>
          <Text>A</Text>
          <Text>B</Text>
        </Box>
      </SimpleGrid>
    </>
  )
}

export default HistroyAI