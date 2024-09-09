import { Flex, Icon, Text } from '@chakra-ui/react'
import { FaCircle } from 'react-icons/fa'

const Lists = ({ name }: { name: string }) => {
  return (
    <Flex align='center'>
      <Icon 
        boxSize={{base: '20px', md: '25px'}} 
        mr={3} 
        as={FaCircle} 
        color='red.300'
      />
      <Text as='b' fontSize={{base: 'md', md: 'xl'}} color='gray.600'>
        {name}
      </Text>
    </Flex>
  )
}

export default Lists